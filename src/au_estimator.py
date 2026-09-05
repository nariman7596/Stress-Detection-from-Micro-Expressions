"""Action Unit intensity estimation (rule-based, FACS-inspired).

Each stress-relevant Action Unit is scored from two independent sources of
evidence, which fail in different ways and therefore complement each other:

* **geometric evidence** -- a landmark distance/ratio compared against the
  subject's own neutral baseline (robust z-score).  Stable, but blind to motion
  smaller than landmark jitter and slow to react.
* **motion evidence** -- coherent residual optical flow in the AU's region, in
  the anatomically correct direction.  Sensitive to 40-200 ms events, but noisy
  and blind to a *held* expression.

Intensities are reported on the FACS A-E scale mapped to ``0..5``.

Everything in this module is a pure function of numbers -- no camera, no
MediaPipe -- which keeps the clinical logic testable.

DE: Regelbasierte AU-Schaetzung aus Geometrie (Baseline-Abweichung) und
Rest-Bewegung (optischer Fluss), Intensitaet 0-5.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .optical_flow import MicroEvent, MicroExpressionDetector, RegionMotion, TemporalSignal

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# AU catalogue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActionUnitSpec:
    """Static description of one Action Unit."""

    code: str
    name: str
    clinical: str
    regions: Tuple[str, ...]


STRESS_ACTION_UNITS: Dict[str, ActionUnitSpec] = {
    "AU1": ActionUnitSpec("AU1", "Inner Brow Raiser", "worry, fear, anticipatory anxiety",
                          ("brow_inner_right", "brow_inner_left")),
    "AU4": ActionUnitSpec("AU4", "Brow Lowerer", "anger, concentration, cognitive load, stress",
                          ("brow_inner_right", "brow_inner_left")),
    "AU6": ActionUnitSpec("AU6", "Cheek Raiser", "genuine (Duchenne) smile; absence marks a masked affect",
                          ("cheek_right", "cheek_left")),
    "AU7": ActionUnitSpec("AU7", "Lid Tightener", "fear, tension, guarding",
                          ("lid_lower_right", "lid_lower_left")),
    "AU12": ActionUnitSpec("AU12", "Lip Corner Puller", "smile; without AU6 a social/masking smile",
                           ("mouth_corner_right", "mouth_corner_left")),
    "AU17": ActionUnitSpec("AU17", "Chin Raiser", "doubt, distress, holding back",
                           ("chin",)),
    "AU20": ActionUnitSpec("AU20", "Lip Stretcher", "fear, apprehension",
                           ("mouth_corner_right", "mouth_corner_left")),
    "AU23": ActionUnitSpec("AU23", "Lip Tightener", "anger, suppression",
                           ("lip_upper", "lip_lower")),
    "AU24": ActionUnitSpec("AU24", "Lip Pressor", "suppression, holding emotion back",
                           ("lip_upper", "lip_lower")),
    "AU43": ActionUnitSpec("AU43", "Eye Closure / Blink", "fatigue, gaze avoidance, withdrawal",
                           ("eye_right", "eye_left")),
}

AU_ORDER: Tuple[str, ...] = tuple(STRESS_ACTION_UNITS)

#: Motion channels: signed, anatomically meaningful combinations of region flow.
MOTION_CHANNELS: Tuple[str, ...] = (
    "brow_rise",
    "brow_converge",
    "cheek_rise",
    "lid_rise",
    "corner_rise",
    "mouth_spread",
    "chin_rise",
    "lip_press",
)


@dataclass
class AUActivation:
    """Estimated activation of one Action Unit for one frame."""

    code: str
    name: str
    intensity: float          # 0..5, FACS A-E scale
    geometric: float          # geometric evidence in robust z units
    motion: float             # motion evidence in robust z units
    micro: bool = False       # a micro-expression burst fired in this AU's region
    confidence: float = 1.0   # frame quality (pose, face size, calibration)

    @property
    def active(self) -> bool:
        """FACS "A" (trace) and above."""
        return self.intensity >= 1.0

    @property
    def letter(self) -> str:
        """FACS intensity letter, ``-`` when inactive."""
        if self.intensity < 1.0:
            return "-"
        return "ABCDE"[min(int(self.intensity) - 1, 4)]


@dataclass
class AUFrame:
    """All AU information for a single processed frame."""

    timestamp: float
    activations: Dict[str, AUActivation]
    channels: Dict[str, float] = field(default_factory=dict)
    micro_events: List[MicroEvent] = field(default_factory=list)
    calibrated: bool = False
    calibration_progress: float = 0.0
    quality: float = 1.0
    eyes_closed: bool = False
    blink_rate: float = 0.0
    micro_rate: float = 0.0

    def intensity(self, code: str) -> float:
        activation = self.activations.get(code)
        return activation.intensity if activation else 0.0

    @property
    def masked_smile(self) -> bool:
        """AU12 without AU6 -- the classic non-Duchenne ("social") smile.

        Clinically interesting: a patient smiling while suppressing distress
        typically produces the lip corner pull without the orbicularis oculi
        contraction that a felt smile carries.
        """
        return self.intensity("AU12") >= 1.5 and self.intensity("AU6") < 0.8


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
class BaselineTracker:
    """Per-subject neutral baseline with robust statistics and slow adaptation.

    Everything downstream is expressed as a deviation from *this* subject's
    neutral face: absolute landmark distances differ far more between people than
    between expressions, so a population baseline would be meaningless.

    After calibration the baseline keeps adapting, but only on frames the
    estimator considers quiet -- otherwise a sustained expression would slowly be
    absorbed into "neutral" and disappear.
    """

    def __init__(
        self,
        calibration_seconds: float = 3.0,
        min_samples: int = 20,
        adapt_tau: float = 90.0,
        relative_sigma_floor: float = 0.02,
        absolute_sigma_floor: float = 1e-3,
    ) -> None:
        self.calibration_seconds = float(calibration_seconds)
        self.min_samples = int(min_samples)
        self.adapt_tau = float(adapt_tau)
        self.relative_sigma_floor = float(relative_sigma_floor)
        self.absolute_sigma_floor = float(absolute_sigma_floor)

        self._buffer: Dict[str, List[float]] = {}
        self._median: Dict[str, float] = {}
        self._sigma: Dict[str, float] = {}
        self._start: Optional[float] = None
        self._last: Optional[float] = None
        self.ready = False

    # ------------------------------------------------------------------ state
    @property
    def progress(self) -> float:
        """Calibration completion in ``[0, 1]``."""
        if self.ready:
            return 1.0
        if self._start is None:
            return 0.0
        samples = len(next(iter(self._buffer.values()), []))
        by_time = (self._last - self._start) / self.calibration_seconds if self._last else 0.0
        by_count = samples / self.min_samples
        return float(np.clip(min(by_time, by_count), 0.0, 1.0))

    def median(self, name: str, default: float = 0.0) -> float:
        return self._median.get(name, default)

    def sigma(self, name: str, default: float = 1.0) -> float:
        return self._sigma.get(name, default)

    # ----------------------------------------------------------------- update
    def observe(self, features: Dict[str, float], timestamp: float, quiet: bool = True) -> None:
        """Feed one frame of features into the baseline."""
        previous = self._last
        if self._start is None:
            self._start = timestamp
        self._last = timestamp

        if not self.ready:
            for name, value in features.items():
                self._buffer.setdefault(name, []).append(float(value))
            samples = len(next(iter(self._buffer.values()), []))
            if timestamp - self._start >= self.calibration_seconds and samples >= self.min_samples:
                self._finalise()
            return

        if not quiet:
            return
        # Exponential adaptation with time constant ``adapt_tau``.
        dt = max(timestamp - previous, 0.0) if previous is not None else 0.0
        alpha = 1.0 - float(np.exp(-max(dt, 1e-3) / self.adapt_tau))
        for name, value in features.items():
            if name in self._median:
                self._median[name] += alpha * (float(value) - self._median[name])

    def _finalise(self) -> None:
        for name, values in self._buffer.items():
            array = np.asarray(values, dtype=np.float64)
            median = float(np.median(array))
            mad = float(np.median(np.abs(array - median)))
            sigma = 1.4826 * mad
            floor = max(self.relative_sigma_floor * abs(median), self.absolute_sigma_floor)
            self._median[name] = median
            self._sigma[name] = max(sigma, floor)
        self._buffer.clear()
        self.ready = True
        logger.info("Baseline calibrated on %d features", len(self._median))

    def reset(self) -> None:
        self.__init__(  # type: ignore[misc]
            self.calibration_seconds,
            self.min_samples,
            self.adapt_tau,
            self.relative_sigma_floor,
            self.absolute_sigma_floor,
        )

    # --------------------------------------------------------------- scoring
    def zscore(self, name: str, value: float) -> float:
        """Robust deviation of ``value`` from the subject's neutral baseline."""
        if not self.ready or name not in self._median:
            return 0.0
        return float((value - self._median[name]) / self._sigma[name])

    def zscores(self, features: Dict[str, float]) -> Dict[str, float]:
        return {name: self.zscore(name, value) for name, value in features.items()}


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
def _positive(value: float) -> float:
    return float(max(value, 0.0))


def evidence_to_intensity(evidence: float, onset: float = 1.5, saturation: float = 6.0) -> float:
    """Map combined evidence (robust z units) onto the FACS 0..5 intensity scale.

    Below ``onset`` nothing is reported (that band is subject jitter); at
    ``saturation`` the AU is scored E.  Linear in between -- deliberately simple
    and inspectable, since no labelled data has been used to fit anything.
    """
    if saturation <= onset:
        raise ValueError("saturation must exceed onset")
    return float(np.clip((evidence - onset) / (saturation - onset), 0.0, 1.0) * 5.0)


class ActionUnitEstimator:
    """Turn landmark geometry + region flow into per-AU intensities.

    Parameters
    ----------
    geometric_weight, motion_weight:
        Relative contribution of the two evidence streams.
    micro_gain:
        Extra evidence added to an AU while a micro-expression burst is open in
        one of its regions -- this is what makes a 3-frame leak visible at all.
    motion_floor, min_motion:
        Noise floors for the flow channels, in canonical-crop pixels at 256 px.
        Without them a perfectly still subject produces enormous z-scores from
        pure flow noise.
    onset, saturation:
        Thresholds of :func:`evidence_to_intensity`.
    """

    def __init__(
        self,
        calibration_seconds: float = 3.0,
        geometric_weight: float = 0.6,
        motion_weight: float = 0.4,
        micro_gain: float = 1.2,
        onset: float = 1.5,
        saturation: float = 6.0,
        blink_closure_ratio: float = 0.6,
        max_blink_duration: float = 0.5,
        micro_window: float = 0.6,
        motion_floor: float = 0.03,
        min_motion: float = 0.35,
    ) -> None:
        self.baseline = BaselineTracker(calibration_seconds=calibration_seconds)
        self.geometric_weight = float(geometric_weight)
        self.motion_weight = float(motion_weight)
        self.micro_gain = float(micro_gain)
        self.onset = float(onset)
        self.saturation = float(saturation)
        self.blink_closure_ratio = float(blink_closure_ratio)
        self.max_blink_duration = float(max_blink_duration)
        self.micro_window = float(micro_window)
        # Motion thresholds are in canonical-crop pixels; see MicroExpressionDetector.
        self.motion_floor = float(motion_floor)

        self._channel_signals: Dict[str, TemporalSignal] = {
            name: TemporalSignal(history_seconds=2.0) for name in MOTION_CHANNELS
        }
        self.micro_detector = MicroExpressionDetector(
            regions=[region for spec in STRESS_ACTION_UNITS.values() for region in spec.regions],
            min_motion=min_motion,
            sigma_floor=motion_floor,
        )
        self._blinks: Deque[float] = deque(maxlen=256)
        self._eyes_closed = False
        self._closure_start: Optional[float] = None
        self._last_frame: Optional[AUFrame] = None

    # ------------------------------------------------------------- channels
    @staticmethod
    def motion_channels(motions: Dict[str, RegionMotion]) -> Dict[str, float]:
        """Combine region flow into signed, anatomically meaningful channels.

        Signs are defined in the *canonical crop*, where the ``_right`` regions
        always sit in the left half of the image, so they hold whether or not the
        preview is mirrored.
        """

        def get(name: str) -> RegionMotion:
            return motions.get(name, RegionMotion(name, 0.0, 0.0, 0.0, 0.0, 0.0))

        brow_r, brow_l = get("brow_inner_right"), get("brow_inner_left")
        cheek_r, cheek_l = get("cheek_right"), get("cheek_left")
        lid_r, lid_l = get("lid_lower_right"), get("lid_lower_left")
        corner_r, corner_l = get("mouth_corner_right"), get("mouth_corner_left")
        lip_up, lip_low = get("lip_upper"), get("lip_lower")
        chin = get("chin")

        return {
            "brow_rise": (brow_r.rise + brow_l.rise) / 2.0,
            # Positive when the brows are drawn towards the midline (AU4).
            "brow_converge": brow_r.dx - brow_l.dx,
            "cheek_rise": (cheek_r.rise + cheek_l.rise) / 2.0,
            "lid_rise": (lid_r.rise + lid_l.rise) / 2.0,
            "corner_rise": (corner_r.rise + corner_l.rise) / 2.0,
            # Positive when the mouth corners move apart (AU12 / AU20).
            "mouth_spread": corner_l.dx - corner_r.dx,
            "chin_rise": chin.rise,
            # Positive when the lips compress vertically (AU23 / AU24).
            "lip_press": lip_up.dy - lip_low.dy,
        }

    def _channel_z(self, channels: Dict[str, float], timestamp: float) -> Dict[str, float]:
        """Robust z-score of each motion channel against its own recent history."""
        scores: Dict[str, float] = {}
        for name, value in channels.items():
            signal = self._channel_signals.setdefault(name, TemporalSignal(history_seconds=2.0))
            scores[name] = signal.zscore(value, floor=self.motion_floor) if len(signal) >= 8 else 0.0
            signal.append(timestamp, value)
        return scores

    # ------------------------------------------------------------------ quality
    @staticmethod
    def frame_quality(features: Dict[str, float]) -> float:
        """Confidence in ``[0, 1]`` from head pose and face size.

        Out-of-plane rotation foreshortens the very displacements we measure, and
        a small face leaves too few pixels for meaningful flow.
        """
        yaw = abs(features.get("yaw_ratio", 0.0))
        roll = abs(features.get("roll_degrees", 0.0))
        size = features.get("interocular_px", 0.0)
        pose = float(np.clip(1.0 - yaw / 0.6, 0.0, 1.0)) * float(np.clip(1.0 - roll / 40.0, 0.0, 1.0))
        resolution = float(np.clip((size - 30.0) / 40.0, 0.0, 1.0))
        return float(np.clip(pose * resolution, 0.0, 1.0))

    # ------------------------------------------------------------------ blinks
    def _update_blinks(self, features: Dict[str, float], timestamp: float) -> Tuple[bool, float, float]:
        """Track eye closure; return ``(eyes_closed, closure_ratio, blinks/min)``."""
        aperture = features.get("eye_aperture", 0.0)
        neutral = self.baseline.median("eye_aperture", aperture)
        closure = 0.0 if neutral <= 1e-6 else float(np.clip(1.0 - aperture / neutral, 0.0, 1.0))

        closed = closure >= self.blink_closure_ratio
        if closed and not self._eyes_closed:
            self._closure_start = timestamp
        elif not closed and self._eyes_closed and self._closure_start is not None:
            if timestamp - self._closure_start <= self.max_blink_duration:
                self._blinks.append(timestamp)
            self._closure_start = None
        self._eyes_closed = closed

        window = 60.0
        while self._blinks and timestamp - self._blinks[0] > window:
            self._blinks.popleft()
        elapsed = min(window, timestamp - (self._blinks[0] if self._blinks else timestamp) + 1e-6)
        rate = len(self._blinks) * 60.0 / max(elapsed, 5.0)
        return closed, closure, float(rate)

    # ------------------------------------------------------------------ update
    def update(
        self,
        features: Dict[str, float],
        motions: Dict[str, RegionMotion],
        timestamp: float,
    ) -> AUFrame:
        """Estimate every stress-relevant AU for one frame.

        ``features`` comes from :func:`src.face_mesh.geometric_features`,
        ``motions`` from :func:`src.optical_flow.aggregate_regions`.
        """
        quality = self.frame_quality(features)

        quiet = True
        if self._last_frame is not None:
            quiet = max((a.intensity for a in self._last_frame.activations.values()), default=0.0) < 1.0
        self.baseline.observe(features, timestamp, quiet=quiet)

        z = self.baseline.zscores(features)
        channels = self.motion_channels(motions)
        channel_z = self._channel_z(channels, timestamp)
        events = self.micro_detector.update(motions, timestamp)
        eyes_closed, closure, blink_rate = self._update_blinks(features, timestamp)

        recent = self.micro_detector.recent_events(timestamp, window=self.micro_window)
        micro_regions = {event.region for event in recent}

        geometric = self._geometric_evidence(z)
        motion = self._motion_evidence(channel_z)

        activations: Dict[str, AUActivation] = {}
        for code, spec in STRESS_ACTION_UNITS.items():
            micro = bool(micro_regions.intersection(spec.regions))
            g, m = geometric.get(code, 0.0), motion.get(code, 0.0)
            evidence = self.geometric_weight * g + self.motion_weight * m
            if micro:
                evidence += self.micro_gain
            intensity = evidence_to_intensity(evidence, self.onset, self.saturation) * quality
            activations[code] = AUActivation(
                code=code,
                name=spec.name,
                intensity=float(intensity),
                geometric=float(g),
                motion=float(m),
                micro=micro,
                confidence=quality,
            )

        # AU43 is measured directly from the aperture rather than through the
        # generic evidence path: closure is unambiguous and needs no threshold.
        activations["AU43"] = AUActivation(
            code="AU43",
            name=STRESS_ACTION_UNITS["AU43"].name,
            intensity=float(np.clip(closure, 0.0, 1.0) * 5.0 * quality),
            geometric=float(-z.get("eye_aperture", 0.0)),
            motion=float(_positive(channel_z.get("lid_rise", 0.0))),
            micro=bool(micro_regions.intersection(STRESS_ACTION_UNITS["AU43"].regions)),
            confidence=quality,
        )

        frame = AUFrame(
            timestamp=timestamp,
            activations=activations,
            channels=channel_z,
            micro_events=events,
            calibrated=self.baseline.ready,
            calibration_progress=self.baseline.progress,
            quality=quality,
            eyes_closed=eyes_closed,
            blink_rate=blink_rate,
            micro_rate=self.micro_detector.event_rate(timestamp),
        )
        self._last_frame = frame
        return frame

    # -------------------------------------------------------------- FACS rules
    @staticmethod
    def _geometric_evidence(z: Dict[str, float]) -> Dict[str, float]:
        """Map baseline deviations onto AU evidence, following FACS descriptions."""
        get = z.get
        cheek_rise = -get("cheek_lid_distance", 0.0)
        lid_tighten = -get("lower_lid_gap", 0.0)
        corner_up = get("corner_elevation", 0.0)
        widen = get("mouth_width", 0.0)

        return {
            # Inner brow up, outer brow comparatively still.
            "AU1": get("inner_brow_raise", 0.0) - 0.3 * _positive(get("outer_brow_raise", 0.0)),
            # Brow closer to the lid AND drawn towards the midline.
            "AU4": 0.6 * -get("brow_lid_distance", 0.0) + 0.4 * -get("brow_separation", 0.0),
            # Cheek travels up towards the lower lid, narrowing the aperture.
            "AU6": 0.7 * cheek_rise + 0.3 * -get("eye_aperture", 0.0),
            # AU6 and AU7 both narrow the eye; only AU6 raises the cheek, so the
            # cheek term is subtracted to keep AU7 specific.
            "AU7": 0.7 * lid_tighten + 0.3 * -get("eye_aperture", 0.0) - 0.5 * _positive(cheek_rise),
            # Corners up and out.
            "AU12": 0.6 * corner_up + 0.4 * widen,
            # Chin boss pushed up, lower lip riding with it.
            "AU17": 0.6 * -get("chin_lip_distance", 0.0) + 0.4 * -get("chin_boss_distance", 0.0),
            # Horizontal stretch *without* elevation is what separates 20 from 12.
            "AU20": widen - 0.8 * _positive(corner_up),
            # Lips thin out and narrow slightly.
            "AU23": 0.7 * -get("lip_thickness", 0.0) + 0.3 * -get("mouth_width", 0.0),
            # Thinning plus a closed mouth: pressing, not just tightening.
            "AU24": 0.5 * -get("lip_thickness", 0.0) + 0.5 * -get("mouth_opening", 0.0),
        }

    @staticmethod
    def _motion_evidence(cz: Dict[str, float]) -> Dict[str, float]:
        """Directional flow evidence per AU (only anatomically correct signs count)."""
        get = cz.get
        return {
            "AU1": _positive(get("brow_rise", 0.0)),
            "AU4": 0.5 * _positive(-get("brow_rise", 0.0)) + 0.5 * _positive(get("brow_converge", 0.0)),
            "AU6": _positive(get("cheek_rise", 0.0)),
            "AU7": _positive(get("lid_rise", 0.0)),
            "AU12": 0.6 * _positive(get("corner_rise", 0.0)) + 0.4 * _positive(get("mouth_spread", 0.0)),
            "AU17": _positive(get("chin_rise", 0.0)),
            "AU20": _positive(get("mouth_spread", 0.0)) - 0.5 * _positive(get("corner_rise", 0.0)),
            "AU23": 0.5 * _positive(get("lip_press", 0.0)) + 0.5 * _positive(-get("mouth_spread", 0.0)),
            "AU24": _positive(get("lip_press", 0.0)),
        }
