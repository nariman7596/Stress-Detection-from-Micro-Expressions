"""Action Units -> a single stress index on a 0-10 scale.

The mapping is a transparent weighted sum, not a learned model.  That is a
deliberate choice for a clinically motivated tool: every point of the score can
be traced back to a named Action Unit, and the weights encode published FACS
associations rather than a fit to unlabelled webcam data.

Contributions
-------------
* **Tonic AU load** -- the weighted mean of the stress-relevant AU intensities.
  AU6 (Duchenne marker) carries a *negative* weight: genuine positive affect
  argues against distress.
* **Masked-smile penalty** -- AU12 without AU6.  A smile that never reaches the
  eyes, while other stress AUs are active, is the signature of concealed affect.
* **Micro-expression rate** -- how often brief leaks fire, independent of their
  content.  Frequent leakage is itself a marker of suppression effort.
* **Blink rate** -- elevated spontaneous blink rate accompanies cognitive load
  and anxiety.

DE: Gewichtete Kombination der Stress-AUs zu einem Index 0-10, plus Zuschlaege
fuer maskiertes Laecheln, Mikro-Ausdrucks-Rate und Blinzelrate.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .au_estimator import AUFrame

logger = logging.getLogger(__name__)

#: Weight of each AU in the tonic load term.  Negative = argues against stress.
AU_STRESS_WEIGHTS: Dict[str, float] = {
    "AU4": 1.00,   # corrugator activity -- the most reported facial stress marker
    "AU7": 0.80,
    "AU24": 0.80,  # suppression
    "AU1": 0.70,   # worry / fear
    "AU23": 0.70,
    "AU17": 0.60,
    "AU20": 0.60,  # fear
    "AU43": 0.30,  # fatigue / avoidance
    "AU12": -0.20,  # a smile, on its own, is weak evidence against stress
    "AU6": -0.40,  # Duchenne marker: genuine positive affect
}

#: Score bands.  Upper bound (exclusive) -> label.
STRESS_LEVELS: Tuple[Tuple[float, str], ...] = (
    (2.0, "calm"),
    (4.0, "mild"),
    (6.0, "moderate"),
    (8.0, "elevated"),
    (10.01, "high"),
)

#: Blink rate (per minute) above which cognitive load / anxiety is suggested.
BLINK_RATE_NEUTRAL = 20.0
BLINK_RATE_SATURATION = 45.0

#: Micro-expression rate (events per minute) at which that term saturates.
MICRO_RATE_SATURATION = 20.0


def stress_level(score: float) -> str:
    """Map a 0-10 score onto a human-readable band."""
    for upper, label in STRESS_LEVELS:
        if score < upper:
            return label
    return STRESS_LEVELS[-1][1]


@dataclass
class StressState:
    """Stress read-out for one frame."""

    timestamp: float
    score: float                 # smoothed index, 0-10  <- the headline number
    instant: float               # unsmoothed index for the same frame
    level: str
    components: Dict[str, float] = field(default_factory=dict)   # term -> points
    contributions: Dict[str, float] = field(default_factory=dict)  # AU -> points
    masked_smile: bool = False
    micro_rate: float = 0.0
    blink_rate: float = 0.0
    quality: float = 1.0
    calibrated: bool = False

    @property
    def top_drivers(self) -> List[Tuple[str, float]]:
        """AUs pushing the score up, strongest first."""
        positive = [(code, value) for code, value in self.contributions.items() if value > 0.05]
        return sorted(positive, key=lambda item: item[1], reverse=True)

    def to_row(self) -> Dict[str, float]:
        """Flat dict for CSV logging."""
        row: Dict[str, float] = {
            "timestamp": round(self.timestamp, 3),
            "stress_score": round(self.score, 3),
            "stress_instant": round(self.instant, 3),
            "level": self.level,
            "masked_smile": int(self.masked_smile),
            "micro_rate": round(self.micro_rate, 2),
            "blink_rate": round(self.blink_rate, 2),
            "quality": round(self.quality, 3),
            "calibrated": int(self.calibrated),
        }
        for code, value in self.contributions.items():
            row[f"contrib_{code}"] = round(value, 3)
        return row


class StressScorer:
    """Accumulate :class:`~src.au_estimator.AUFrame` results into a stress index.

    Parameters
    ----------
    smoothing_tau:
        Time constant (seconds) of the exponential smoother.  Micro-expressions
        are brief by definition; the *index* they feed should move on the time
        scale of a clinical impression, not of a single frame.
    masked_smile_penalty, micro_weight, blink_weight:
        Maximum points each modifier can add.
    require_calibration:
        While the baseline is still calibrating the AU z-scores are meaningless,
        so the score is held at 0 and flagged rather than reported as "calm".
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        smoothing_tau: float = 1.5,
        masked_smile_penalty: float = 1.0,
        micro_weight: float = 1.5,
        blink_weight: float = 1.0,
        require_calibration: bool = True,
        history_seconds: float = 120.0,
    ) -> None:
        self.weights = dict(weights or AU_STRESS_WEIGHTS)
        self.smoothing_tau = float(smoothing_tau)
        self.masked_smile_penalty = float(masked_smile_penalty)
        self.micro_weight = float(micro_weight)
        self.blink_weight = float(blink_weight)
        self.require_calibration = require_calibration
        self.history_seconds = float(history_seconds)

        self._positive_mass = sum(w for w in self.weights.values() if w > 0)
        self._score: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self.history: Deque[StressState] = deque(maxlen=4096)

    # ------------------------------------------------------------------ core
    def _tonic_load(self, frame: AUFrame) -> Tuple[float, Dict[str, float]]:
        """Weighted AU load in points (0-10) plus the per-AU breakdown."""
        contributions: Dict[str, float] = {}
        total = 0.0
        for code, weight in self.weights.items():
            normalized = frame.intensity(code) / 5.0  # FACS 0..5 -> 0..1
            points = weight * normalized * 10.0 / self._positive_mass
            contributions[code] = points
            total += points
        return total, contributions

    def update(self, frame: AUFrame) -> StressState:
        """Fold one AU frame into the running stress index."""
        if self.require_calibration and not frame.calibrated:
            # Zero-filled rather than empty, so that every state -- calibrating
            # or scored -- exposes the same schema to CSV logging and plotting.
            state = StressState(
                timestamp=frame.timestamp,
                score=0.0,
                instant=0.0,
                level="calibrating",
                components={key: 0.0 for key in ("au_load", "masked_smile", "micro_rate", "blink_rate")},
                contributions={code: 0.0 for code in self.weights},
                quality=frame.quality,
                calibrated=False,
                micro_rate=frame.micro_rate,
                blink_rate=frame.blink_rate,
            )
            self.history.append(state)
            return state

        load, contributions = self._tonic_load(frame)

        masked = frame.masked_smile
        masked_points = self.masked_smile_penalty if masked else 0.0

        micro_points = self.micro_weight * float(
            np.clip(frame.micro_rate / MICRO_RATE_SATURATION, 0.0, 1.0)
        )
        blink_span = max(BLINK_RATE_SATURATION - BLINK_RATE_NEUTRAL, 1e-6)
        blink_points = self.blink_weight * float(
            np.clip((frame.blink_rate - BLINK_RATE_NEUTRAL) / blink_span, 0.0, 1.0)
        )

        components = {
            "au_load": load,
            "masked_smile": masked_points,
            "micro_rate": micro_points,
            "blink_rate": blink_points,
        }
        instant = float(np.clip(sum(components.values()), 0.0, 10.0))

        dt = 0.0 if self._last_timestamp is None else max(frame.timestamp - self._last_timestamp, 0.0)
        if self._score is None:
            self._score = instant
        else:
            alpha = 1.0 - float(np.exp(-max(dt, 1e-3) / max(self.smoothing_tau, 1e-3)))
            self._score += alpha * (instant - self._score)
        self._last_timestamp = frame.timestamp

        state = StressState(
            timestamp=frame.timestamp,
            score=float(np.clip(self._score, 0.0, 10.0)),
            instant=instant,
            level=stress_level(self._score),
            components=components,
            contributions=contributions,
            masked_smile=masked,
            micro_rate=frame.micro_rate,
            blink_rate=frame.blink_rate,
            quality=frame.quality,
            calibrated=True,
        )
        self.history.append(state)
        self._prune(frame.timestamp)
        return state

    def _prune(self, timestamp: float) -> None:
        cutoff = timestamp - self.history_seconds
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()

    # ------------------------------------------------------------- reporting
    def series(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(timestamps, scores)`` of the retained history, for plotting."""
        times = np.array([s.timestamp for s in self.history], dtype=np.float64)
        scores = np.array([s.score for s in self.history], dtype=np.float64)
        return times, scores

    def summary(self) -> Dict[str, float]:
        """Session-level summary suitable for printing at the end of a run."""
        scored = [s for s in self.history if s.calibrated]
        if not scored:
            return {"frames": 0}
        values = np.array([s.score for s in scored], dtype=np.float64)
        return {
            "frames": float(len(scored)),
            "duration_s": float(scored[-1].timestamp - scored[0].timestamp),
            "mean_score": float(values.mean()),
            "peak_score": float(values.max()),
            "p95_score": float(np.percentile(values, 95)),
            "masked_smile_frames": float(sum(1 for s in scored if s.masked_smile)),
            "mean_micro_rate": float(np.mean([s.micro_rate for s in scored])),
            "mean_blink_rate": float(np.mean([s.blink_rate for s in scored])),
        }

    def reset(self) -> None:
        self._score = None
        self._last_timestamp = None
        self.history.clear()
