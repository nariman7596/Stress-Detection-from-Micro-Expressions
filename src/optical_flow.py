"""Micro-motion extraction with dense optical flow.

Why optical flow at all?  Standard expression recognition classifies a *held*
facial configuration.  Micro-expressions are the opposite: 40-200 ms leaks whose
peak displacement is often a fraction of a landmark's own jitter.  Differentiating
the image in time (Farneback dense flow) is far more sensitive to that regime than
differentiating landmark positions.

The pipeline here is:

    aligned face crop (t-1, t)
        -> Farneback dense flow
        -> subtract the global/rigid component (residual = non-rigid motion)
        -> aggregate per AU region  (:class:`RegionMotion`)
        -> temporal buffering + burst detection (:class:`MicroExpressionDetector`)

DE: Dichter optischer Fluss (Farneback) auf dem normalisierten Gesicht,
Rest-Bewegung pro Region, Burst-Erkennung im Mikro-Zeitfenster.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Micro-expression duration bounds (seconds).  Ekman's classical definition is
#: 1/25 s to 1/5 s; the CASME II / SAMM corpora accept up to ~0.5 s total.
MICRO_MIN_DURATION = 0.04
MICRO_MAX_DURATION = 0.50

#: Temporal band (Hz) that 40-200 ms events occupy.  Slow head drift and lighting
#: changes sit below it, sensor noise above it.
MICRO_BAND_HZ = (2.0, 25.0)


@dataclass
class RegionMotion:
    """Aggregated residual motion inside one AU region for a single frame pair."""

    region: str
    dx: float           # mean horizontal displacement (px per frame pair, + = right)
    dy: float           # mean vertical displacement (+ = down, image convention)
    magnitude: float    # mean |flow| -- how much motion, regardless of direction
    coherence: float    # |mean vector| / mean |flow| in [0, 1]; 1 = one direction
    energy: float       # mean squared magnitude -- emphasises brief strong motion

    @property
    def rise(self) -> float:
        """Upward displacement (positive when the region moves up in the image)."""
        return -self.dy

    @property
    def directed(self) -> float:
        """Coherent motion magnitude -- noise is incoherent and gets suppressed."""
        return self.magnitude * self.coherence


@dataclass
class FlowResult:
    """Dense residual flow for one frame pair, plus its per-region aggregation."""

    flow: np.ndarray                    # (H, W, 2) residual flow
    global_motion: Tuple[float, float]  # the rigid component that was removed
    dt: float                           # seconds between the two frames
    timestamp: float
    regions: Dict[str, RegionMotion] = field(default_factory=dict)


@dataclass
class MicroEvent:
    """A detected micro-expression burst in one region."""

    region: str
    onset: float
    apex: float
    offset: float
    peak_z: float
    peak_value: float

    @property
    def duration(self) -> float:
        return self.offset - self.onset


class MicroMotionExtractor:
    """Farneback dense flow on canonically aligned face crops.

    Parameters
    ----------
    downscale:
        Compute flow on a smaller image for speed.  ``0.5`` on a 256 px crop keeps
        AU-region detail while roughly quartering the cost; displacements are
        rescaled back to crop pixels.
    compensate_global:
        Subtract the median flow vector over the whole face.  The face aligner
        already removes rigid motion in the image plane, but residual out-of-plane
        head motion and warp interpolation leave a slowly varying global field;
        the median is a robust estimate of it and is insensitive to a few regions
        that genuinely move.
    smooth_sigma:
        Gaussian blur applied to the grayscale crop before flow.  Suppresses
        sensor noise, which otherwise dominates at these displacement magnitudes.
    """

    def __init__(
        self,
        downscale: float = 0.5,
        compensate_global: bool = True,
        smooth_sigma: float = 1.0,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ) -> None:
        if not 0.1 <= downscale <= 1.0:
            raise ValueError("downscale must be in [0.1, 1.0]")
        self.downscale = float(downscale)
        self.compensate_global = compensate_global
        self.smooth_sigma = float(smooth_sigma)
        self.farneback_params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0,
        )
        self._previous: Optional[np.ndarray] = None
        self._previous_time: Optional[float] = None

    # ------------------------------------------------------------------ core
    def _prepare(self, image: np.ndarray) -> np.ndarray:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.downscale < 1.0:
            gray = cv2.resize(gray, None, fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_AREA)
        if self.smooth_sigma > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), self.smooth_sigma)
        return gray

    def reset(self) -> None:
        """Forget the previous frame (call after a tracking gap)."""
        self._previous = None
        self._previous_time = None

    def update(self, image: np.ndarray, timestamp: float) -> Optional[FlowResult]:
        """Compute residual flow between the previous crop and ``image``.

        Returns ``None`` for the first frame (no pair yet).
        """
        current = self._prepare(image)
        if self._previous is None or self._previous.shape != current.shape:
            self._previous, self._previous_time = current, timestamp
            return None

        flow = cv2.calcOpticalFlowFarneback(self._previous, current, None, **self.farneback_params)
        if self.downscale < 1.0:
            # Rescale both the field size and the vectors back to crop pixels.
            flow = cv2.resize(flow, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            flow /= self.downscale

        global_motion = (0.0, 0.0)
        if self.compensate_global:
            median = np.median(flow.reshape(-1, 2), axis=0)
            flow = flow - median
            global_motion = (float(median[0]), float(median[1]))

        previous_time = self._previous_time if self._previous_time is not None else timestamp
        dt = max(timestamp - previous_time, 1e-6)
        self._previous, self._previous_time = current, timestamp
        return FlowResult(flow=flow, global_motion=global_motion, dt=dt, timestamp=timestamp)


def region_motion(flow: np.ndarray, box: Tuple[int, int, int, int], region: str = "") -> RegionMotion:
    """Aggregate a dense flow field inside an axis-aligned box."""
    x0, y0, x1, y1 = (int(v) for v in box)
    height, width = flow.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return RegionMotion(region, 0.0, 0.0, 0.0, 0.0, 0.0)

    patch = flow[y0:y1, x0:x1].reshape(-1, 2)
    magnitudes = np.linalg.norm(patch, axis=1)
    mean_vector = patch.mean(axis=0)
    mean_magnitude = float(magnitudes.mean())
    coherence = float(np.linalg.norm(mean_vector) / mean_magnitude) if mean_magnitude > 1e-9 else 0.0
    return RegionMotion(
        region=region,
        dx=float(mean_vector[0]),
        dy=float(mean_vector[1]),
        magnitude=mean_magnitude,
        coherence=min(coherence, 1.0),
        energy=float((magnitudes ** 2).mean()),
    )


def aggregate_regions(
    flow: np.ndarray,
    boxes: Dict[str, Tuple[int, int, int, int]],
) -> Dict[str, RegionMotion]:
    """Aggregate a flow field over every named region box."""
    return {name: region_motion(flow, box, name) for name, box in boxes.items()}


# --------------------------------------------------------------------------- #
# Temporal analysis
# --------------------------------------------------------------------------- #
class TemporalSignal:
    """A short, time-stamped ring buffer with robust (median/MAD) statistics.

    Robust statistics matter here: a burst is exactly the kind of outlier that
    would inflate a mean/std baseline and hide itself.
    """

    def __init__(self, history_seconds: float = 2.0, max_samples: int = 600) -> None:
        self.history_seconds = float(history_seconds)
        self._samples: Deque[Tuple[float, float]] = deque(maxlen=max_samples)

    def append(self, timestamp: float, value: float) -> None:
        self._samples.append((float(timestamp), float(value)))
        cutoff = timestamp - self.history_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def times(self) -> np.ndarray:
        return np.array([t for t, _ in self._samples], dtype=np.float64)

    @property
    def values(self) -> np.ndarray:
        return np.array([v for _, v in self._samples], dtype=np.float64)

    def robust_stats(self) -> Tuple[float, float]:
        """Return ``(median, sigma)`` with sigma estimated from the MAD."""
        values = self.values
        if values.size == 0:
            return 0.0, 0.0
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median, 1.4826 * mad

    def zscore(self, value: float, floor: float = 1e-4) -> float:
        median, sigma = self.robust_stats()
        return (value - median) / max(sigma, floor)


def band_energy_ratio(
    times: np.ndarray,
    values: np.ndarray,
    band: Tuple[float, float] = MICRO_BAND_HZ,
) -> float:
    """Fraction of the signal's spectral energy inside ``band``.

    The signal is resampled onto a uniform grid (frames arrive with jitter),
    linearly detrended and windowed before the FFT.  A high ratio means the
    region's motion is dominated by short bursts rather than slow drift, which is
    the temporal signature of a micro-expression.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.size < 8 or times[-1] - times[0] <= 0:
        return 0.0

    duration = times[-1] - times[0]
    n = int(times.size)
    uniform_t = np.linspace(times[0], times[-1], n)
    signal = np.interp(uniform_t, times, values)
    signal = signal - np.polyval(np.polyfit(uniform_t, signal, 1), uniform_t)
    signal = signal * np.hanning(n)

    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(n, d=duration / (n - 1))
    total = float(spectrum[1:].sum())
    if total <= 0:
        return 0.0
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(spectrum[mask].sum() / total)


class MicroExpressionDetector:
    """Detect short motion bursts per region and classify them by duration.

    A burst starts when the robust z-score of a region's coherent motion exceeds
    ``z_onset`` *and* the motion clears an absolute floor, and it ends when the
    z-score drops below ``z_offset``.  Only bursts whose total duration falls
    inside :data:`MICRO_MIN_DURATION` .. :data:`MICRO_MAX_DURATION` are reported
    as micro-expressions -- longer ones are ordinary (macro) expressions and are
    deliberately discarded.

    The absolute floor is essential in practice.  A subject sitting still gives a
    near-zero MAD, so a purely relative criterion turns the flow's own noise into
    huge z-scores and fires constantly.  Measured on a static portrait clip, the
    residual per-region motion sits near 0.03 px with a 95th percentile around
    0.11 px per frame pair and noise excursions to ~0.4 px; the default
    ``min_motion`` sits above those excursions while still passing an induced
    130 ms brow twitch (see ``notebooks/01_au_exploration.ipynb``).  Lower it if
    the camera is clean and well lit, raise it for a noisy or distant source.

    Parameters
    ----------
    min_motion, sigma_floor:
        Both in pixels of the canonical crop *at 256 px*.  Scale them with the
        crop size (:class:`~src.pipeline.StressPipeline` does this for you).
    """

    def __init__(
        self,
        regions: Iterable[str],
        history_seconds: float = 2.0,
        z_onset: float = 3.0,
        z_offset: float = 1.5,
        min_duration: float = MICRO_MIN_DURATION,
        max_duration: float = MICRO_MAX_DURATION,
        event_memory: float = 10.0,
        min_motion: float = 0.35,
        sigma_floor: float = 0.03,
        cluster_tolerance: float = 0.1,
    ) -> None:
        self.signals: Dict[str, TemporalSignal] = {
            name: TemporalSignal(history_seconds) for name in regions
        }
        self.z_onset = float(z_onset)
        self.z_offset = float(z_offset)
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)
        self.event_memory = float(event_memory)
        self.min_motion = float(min_motion)
        self.sigma_floor = float(sigma_floor)
        self.cluster_tolerance = float(cluster_tolerance)
        self._first_timestamp: Optional[float] = None
        self._active: Dict[str, Dict[str, float]] = {}
        self.events: Deque[MicroEvent] = deque(maxlen=256)

    def update(self, motions: Dict[str, RegionMotion], timestamp: float) -> List[MicroEvent]:
        """Feed one frame of region motions; return events that closed just now."""
        closed: List[MicroEvent] = []
        if self._first_timestamp is None:
            self._first_timestamp = timestamp
        for name, motion in motions.items():
            signal = self.signals.setdefault(name, TemporalSignal())
            value = motion.directed
            z = signal.zscore(value, floor=self.sigma_floor) if len(signal) >= 8 else 0.0
            signal.append(timestamp, value)

            state = self._active.get(name)
            if state is None:
                if z >= self.z_onset and value >= self.min_motion:
                    self._active[name] = {
                        "onset": timestamp,
                        "apex": timestamp,
                        "peak_z": z,
                        "peak_value": value,
                        "macro": 0.0,
                    }
                continue

            if z > state["peak_z"]:
                state.update(peak_z=z, peak_value=value, apex=timestamp)

            if timestamp - state["onset"] > self.max_duration:
                # Sustained -> macro expression.  The burst is kept open (but
                # flagged) until the signal actually falls back, so its tail
                # cannot re-trigger as a spurious micro-expression once the
                # rolling baseline has caught up with the new level.
                state["macro"] = 1.0

            if z < self.z_offset:
                duration = timestamp - state["onset"]
                if not state["macro"] and self.min_duration <= duration <= self.max_duration:
                    event = MicroEvent(
                        region=name,
                        onset=state["onset"],
                        apex=state["apex"],
                        offset=timestamp,
                        peak_z=state["peak_z"],
                        peak_value=state["peak_value"],
                    )
                    self.events.append(event)
                    closed.append(event)
                self._active.pop(name, None)
        self._prune(timestamp)
        return closed

    def _prune(self, timestamp: float) -> None:
        cutoff = timestamp - self.event_memory
        while self.events and self.events[0].offset < cutoff:
            self.events.popleft()

    def recent_events(self, timestamp: float, window: float = 5.0) -> List[MicroEvent]:
        return [e for e in self.events if timestamp - e.offset <= window]

    def clustered_events(self, events: Iterable[MicroEvent]) -> List[List[MicroEvent]]:
        """Group region events that belong to one physical leak.

        A single micro-expression moves several regions at once -- the induced
        brow twitch in the validation clip fires in thirteen of them -- so events
        whose onsets fall within ``cluster_tolerance`` are one event, not
        thirteen.  Counting regions instead would inflate the rate by an order of
        magnitude.
        """
        ordered = sorted(events, key=lambda e: e.onset)
        clusters: List[List[MicroEvent]] = []
        for event in ordered:
            if clusters and event.onset - clusters[-1][0].onset <= self.cluster_tolerance:
                clusters[-1].append(event)
            else:
                clusters.append([event])
        return clusters

    def event_rate(self, timestamp: float, window: float = 10.0) -> float:
        """Micro-expression events per minute over the trailing ``window``.

        Normalised by the time actually observed, so a session shorter than the
        window does not read as a lower rate than it really has.
        """
        window = max(window, 1e-6)
        recent = [e for e in self.events if timestamp - e.offset <= window]
        elapsed = window if self._first_timestamp is None else min(
            window, max(timestamp - self._first_timestamp, 1e-6)
        )
        return len(self.clustered_events(recent)) * 60.0 / elapsed

    def band_ratio(self, region: str, band: Tuple[float, float] = MICRO_BAND_HZ) -> float:
        signal = self.signals.get(region)
        if signal is None or len(signal) < 8:
            return 0.0
        return band_energy_ratio(signal.times, signal.values, band)
