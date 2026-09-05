"""End-to-end pipeline behaviour, with a stub landmark detector.

MediaPipe (and, on 1.x, a downloaded model bundle) is not required here: the
detector is injected, so every stage after it -- alignment, optical flow, AU
estimation and scoring -- runs on synthetic frames exactly as it does live.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pytest

from src.face_mesh import FACE_REGIONS, FaceLandmarks
from src.pipeline import StressPipeline

from conftest import make_face


class StubDetector:
    """Stands in for :class:`~src.face_mesh.FaceMeshDetector`."""

    backend = "stub"

    def __init__(self, landmarks_for=None, fail_after: Optional[int] = None) -> None:
        self._landmarks_for = landmarks_for or (lambda index: make_face())
        self._fail_after = fail_after
        self.calls = 0
        self.closed = False

    def process(self, frame_bgr, timestamp_ms=None) -> Optional[FaceLandmarks]:
        index = self.calls
        self.calls += 1
        if self._fail_after is not None and index >= self._fail_after:
            return None
        return self._landmarks_for(index)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def moving_frames():
    """A textured 'face' that drifts slightly -- enough to exercise the flow stage."""
    rng = np.random.default_rng(11)
    base = cv2.GaussianBlur((rng.random((480, 640)) * 255).astype(np.uint8), (0, 0), 2.0)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    def frame(index: int) -> np.ndarray:
        matrix = np.float32([[1, 0, 0.2 * np.sin(index / 7.0)], [0, 1, 0.2 * np.cos(index / 9.0)]])
        return cv2.warpAffine(base, matrix, (640, 480), borderMode=cv2.BORDER_REFLECT)

    return frame


def make_pipeline(detector, **kwargs) -> StressPipeline:
    return StressPipeline(calibration_seconds=1.0, align_size=128, detector=detector, **kwargs)


def test_no_face_yields_an_empty_result(moving_frames):
    pipeline = make_pipeline(StubDetector(fail_after=0))
    result = pipeline.process(moving_frames(0), 0.0)
    assert not result.face_found
    assert result.au_frame is None and result.stress is None


def test_first_frame_has_landmarks_but_no_flow(moving_frames):
    pipeline = make_pipeline(StubDetector())
    result = pipeline.process(moving_frames(0), 0.0)
    assert result.face_found
    assert result.aligned is not None
    assert result.flow is None      # a flow field needs two frames
    assert result.au_frame is not None
    assert result.stress.level == "calibrating"


def test_region_boxes_cover_every_region(moving_frames):
    pipeline = make_pipeline(StubDetector())
    result = pipeline.process(moving_frames(0), 0.0)
    boxes = pipeline.region_boxes(result.aligned.landmarks)
    assert set(boxes) == set(FACE_REGIONS)
    for x0, y0, x1, y1 in boxes.values():
        assert 0 <= x0 < x1 <= 128
        assert 0 <= y0 < y1 <= 128


def test_pipeline_calibrates_and_then_scores(moving_frames):
    pipeline = make_pipeline(StubDetector())
    fps = 30.0
    states = [pipeline.process(moving_frames(i), i / fps).stress for i in range(90)]

    assert states[0].level == "calibrating"
    assert not states[0].calibrated
    assert states[-1].calibrated
    assert 0.0 <= states[-1].score <= 10.0
    assert pipeline.estimator.baseline.ready


def test_flow_regions_are_populated_after_the_second_frame(moving_frames):
    pipeline = make_pipeline(StubDetector())
    pipeline.process(moving_frames(0), 0.0)
    result = pipeline.process(moving_frames(1), 1 / 30)
    assert result.flow is not None
    assert set(result.motions) == set(FACE_REGIONS)
    assert result.flow.regions is result.motions


def test_a_still_neutral_face_stays_calm(moving_frames):
    pipeline = make_pipeline(StubDetector())
    fps = 30.0
    state = None
    for index in range(150):
        state = pipeline.process(moving_frames(index), index / fps).stress
    assert state.calibrated
    assert state.score < 4.0, "a neutral, barely-moving face must not read as stressed"


def test_tracking_loss_resets_the_flow_reference(moving_frames):
    """After a gap the next flow pair would straddle it and fake a burst."""
    pipeline = make_pipeline(StubDetector(fail_after=3))
    for index in range(3):
        pipeline.process(moving_frames(index), index / 30.0)
    assert pipeline.motion._previous is not None

    pipeline.process(moving_frames(3), 3 / 30.0)   # first miss: keep the reference
    assert pipeline.motion._previous is not None
    pipeline.process(moving_frames(4), 4 / 30.0)   # second miss: drop it
    assert pipeline.motion._previous is None


def test_expression_raises_the_score_above_neutral(moving_frames):
    """A held stress configuration must score above the subject's own baseline."""
    fps = 30.0

    def landmarks_for(index: int) -> FaceLandmarks:
        if index < 60:  # calibration: neutral
            return make_face(seed=index)
        # brows lowered and drawn together (AU4), lids tightened (AU7),
        # lips pressed thin (AU24) -- a classic stress configuration.
        return make_face(seed=index, brow_height=9.0, brow_separation=17.0,
                         lower_lid=3.5, lip_thickness=3.0)

    pipeline = make_pipeline(StubDetector(landmarks_for=landmarks_for))
    neutral_state = None
    for index in range(120):
        state = pipeline.process(moving_frames(index), index / fps).stress
        if index == 59:
            neutral_state = state
    assert neutral_state.score < state.score
    assert state.score > 2.0
    assert state.top_drivers


def test_close_releases_the_detector(moving_frames):
    detector = StubDetector()
    pipeline = make_pipeline(detector)
    with pipeline:
        pipeline.process(moving_frames(0), 0.0)
    assert detector.closed


def test_micro_sensitivity_scales_the_noise_gate():
    strict = make_pipeline(StubDetector(), micro_sensitivity=0.5)
    loose = make_pipeline(StubDetector(), micro_sensitivity=2.0)
    assert strict.estimator.micro_detector.min_motion > loose.estimator.micro_detector.min_motion


def test_motion_thresholds_scale_with_the_crop_size():
    small = make_pipeline(StubDetector())                       # align_size=128
    large = StressPipeline(calibration_seconds=1.0, align_size=512, detector=StubDetector())
    assert large.estimator.micro_detector.min_motion == pytest.approx(
        4.0 * small.estimator.micro_detector.min_motion
    )
