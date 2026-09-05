"""End-to-end per-frame pipeline.

Wiring order, and why:

1. **FaceMesh** on the full frame -> landmarks in image pixels.
2. **Canonical alignment** -> a fixed-size crop in which the eyes and nose base
   always land on the same pixels.  Without this step the optical flow field is
   dominated by head motion and camera distance.
3. **Farneback flow** between consecutive *aligned* crops, minus its global
   component -> residual, non-rigid facial motion.
4. **Region aggregation** over the AU regions, in crop coordinates.
5. **AU estimation** from baseline-relative geometry + directional motion.
6. **Stress scoring** -> a smoothed 0-10 index.

DE: Verkettung von Landmarks -> Normalisierung -> Fluss -> AUs -> Stress-Index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .au_estimator import ActionUnitEstimator, AUFrame
from .face_mesh import (
    FACE_REGIONS,
    AlignedFace,
    FaceAligner,
    FaceLandmarks,
    FaceMeshDetector,
    geometric_features,
)
from .optical_flow import FlowResult, MicroMotionExtractor, RegionMotion, aggregate_regions
from .stress_scorer import StressScorer, StressState

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything one processed frame produced."""

    timestamp: float
    landmarks: Optional[FaceLandmarks] = None
    aligned: Optional[AlignedFace] = None
    flow: Optional[FlowResult] = None
    motions: Dict[str, RegionMotion] = field(default_factory=dict)
    au_frame: Optional[AUFrame] = None
    stress: Optional[StressState] = None

    @property
    def face_found(self) -> bool:
        return self.landmarks is not None


class StressPipeline:
    """Own every stage and expose a single :meth:`process` call per frame.

    Parameters
    ----------
    calibration_seconds:
        Length of the neutral-baseline calibration.  The subject should hold a
        relaxed, forward-facing expression for this long at the start.
    align_size:
        Side length of the canonical face crop.  256 px keeps AU regions large
        enough for Farneback while staying real-time on Apple Silicon.
    region_padding:
        Relative expansion of each AU region's flow box.
    micro_sensitivity:
        Scales the micro-expression noise gate.  Above 1.0 detects fainter leaks
        at the price of more false positives; below 1.0 is stricter.
    detector:
        Pre-built landmark detector; a default :class:`FaceMeshDetector` is
        created when omitted.
    """

    def __init__(
        self,
        calibration_seconds: float = 3.0,
        align_size: int = 256,
        region_padding: float = 0.15,
        flow_downscale: float = 0.5,
        micro_sensitivity: float = 1.0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
        model_asset_path: Optional[str] = None,
        smoothing_tau: float = 1.5,
        detector: Optional[FaceMeshDetector] = None,
    ) -> None:
        # ``detector`` is injectable so the pipeline can be exercised in tests
        # without MediaPipe (and its model download) being present.
        self.detector = detector or FaceMeshDetector(
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_asset_path=model_asset_path,
        )
        self.aligner = FaceAligner(size=align_size)
        self.motion = MicroMotionExtractor(downscale=flow_downscale)
        # Flow thresholds are expressed in pixels of a 256 px canonical crop;
        # a different crop size scales every displacement with it.
        motion_scale = align_size / 256.0 / max(micro_sensitivity, 1e-3)
        self.estimator = ActionUnitEstimator(
            calibration_seconds=calibration_seconds,
            motion_floor=0.03 * motion_scale,
            min_motion=0.35 * motion_scale,
        )
        self.scorer = StressScorer(smoothing_tau=smoothing_tau)
        self.region_padding = float(region_padding)
        self._missed_frames = 0

    # ------------------------------------------------------------------ core
    def region_boxes(self, landmarks: FaceLandmarks) -> Dict[str, Tuple[int, int, int, int]]:
        """Flow boxes for every AU region, in aligned-crop coordinates."""
        return {name: landmarks.region_box(name, self.region_padding) for name in FACE_REGIONS}

    def process(self, image: np.ndarray, timestamp: float) -> PipelineResult:
        """Run the whole pipeline on one BGR frame."""
        landmarks = self.detector.process(image, timestamp_ms=int(timestamp * 1000))
        if landmarks is None:
            self._missed_frames += 1
            if self._missed_frames >= 2:
                # Two consecutive misses: the next flow pair would straddle a gap
                # and produce a spurious burst, so drop the reference frame.
                self.motion.reset()
            return PipelineResult(timestamp=timestamp)
        self._missed_frames = 0

        aligned = self.aligner.align(image, landmarks)
        features = geometric_features(aligned.landmarks)

        # ``MicroMotionExtractor`` handles the greyscale conversion itself.
        flow = self.motion.update(aligned.image, timestamp)

        motions: Dict[str, RegionMotion] = {}
        if flow is not None:
            motions = aggregate_regions(flow.flow, self.region_boxes(aligned.landmarks))
            flow.regions = motions

        au_frame = self.estimator.update(features, motions, timestamp)
        stress = self.scorer.update(au_frame)

        return PipelineResult(
            timestamp=timestamp,
            landmarks=landmarks,
            aligned=aligned,
            flow=flow,
            motions=motions,
            au_frame=au_frame,
            stress=stress,
        )

    # -------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self.detector.close()

    def __enter__(self) -> "StressPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
