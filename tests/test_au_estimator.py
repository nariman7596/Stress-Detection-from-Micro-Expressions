"""Action Unit estimation: baseline, FACS rules, blinks and quality gating."""

from __future__ import annotations

import numpy as np
import pytest

from src.au_estimator import (
    AU_ORDER,
    STRESS_ACTION_UNITS,
    ActionUnitEstimator,
    AUActivation,
    BaselineTracker,
    evidence_to_intensity,
)
from src.face_mesh import geometric_features
from src.optical_flow import RegionMotion

from conftest import make_face

FPS = 30.0
NO_MOTION: dict = {}


def calibrate(estimator: ActionUnitEstimator, seconds: float = 4.0, **face_kwargs):
    """Feed a neutral face until the personal baseline is ready."""
    frames = int(seconds * FPS)
    frame = None
    for index in range(frames):
        features = geometric_features(make_face(seed=index, **face_kwargs))
        frame = estimator.update(features, NO_MOTION, index / FPS)
    return frame


def motions_for(regions, dx=0.0, dy=0.0, magnitude=None, coherence=1.0):
    magnitude = magnitude if magnitude is not None else float(np.hypot(dx, dy))
    return {r: RegionMotion(r, dx, dy, magnitude, coherence, magnitude ** 2) for r in regions}


# --------------------------------------------------------------------------- #
# Intensity mapping
# --------------------------------------------------------------------------- #
def test_evidence_below_onset_is_silent():
    assert evidence_to_intensity(0.0) == 0.0
    assert evidence_to_intensity(1.4) == 0.0


def test_evidence_saturates_at_five():
    assert evidence_to_intensity(6.0) == pytest.approx(5.0)
    assert evidence_to_intensity(50.0) == pytest.approx(5.0)
    assert 0.0 < evidence_to_intensity(3.0) < 5.0


def test_evidence_mapping_is_monotonic():
    values = [evidence_to_intensity(e) for e in np.linspace(0, 8, 40)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        evidence_to_intensity(1.0, onset=3.0, saturation=3.0)


def test_facs_letter_scale():
    assert AUActivation("AU4", "Brow Lowerer", 0.4, 0, 0).letter == "-"
    assert AUActivation("AU4", "Brow Lowerer", 0.4, 0, 0).active is False
    assert AUActivation("AU4", "Brow Lowerer", 1.2, 0, 0).letter == "A"
    assert AUActivation("AU4", "Brow Lowerer", 5.0, 0, 0).letter == "E"
    assert AUActivation("AU4", "Brow Lowerer", 5.0, 0, 0).active is True


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def test_baseline_becomes_ready_and_reports_progress():
    tracker = BaselineTracker(calibration_seconds=1.0, min_samples=10)
    assert tracker.progress == 0.0
    for index in range(40):
        tracker.observe({"x": 1.0 + 0.01 * (index % 3)}, index / FPS)
    assert tracker.ready
    assert tracker.progress == 1.0
    assert tracker.median("x") == pytest.approx(1.0, abs=0.02)


def test_baseline_zscore_is_zero_before_calibration():
    tracker = BaselineTracker(calibration_seconds=10.0)
    tracker.observe({"x": 1.0}, 0.0)
    assert tracker.zscore("x", 99.0) == 0.0


def test_baseline_sigma_floor_prevents_blow_up():
    """A perfectly still subject must not turn sensor noise into huge z-scores."""
    tracker = BaselineTracker(calibration_seconds=1.0, min_samples=10)
    for index in range(60):
        tracker.observe({"x": 1.0}, index / FPS)  # zero variance
    assert tracker.sigma("x") >= 0.02
    assert abs(tracker.zscore("x", 1.001)) < 1.0


def test_baseline_adapts_slowly_on_quiet_frames():
    tracker = BaselineTracker(calibration_seconds=1.0, min_samples=10, adapt_tau=5.0)
    for index in range(60):
        tracker.observe({"x": 1.0}, index / FPS)
    start = tracker.median("x")
    for index in range(60, 300):
        tracker.observe({"x": 2.0}, index / FPS, quiet=True)
    drifted = tracker.median("x")
    assert start < drifted < 2.0  # moved towards the new level, but not instantly


def test_baseline_does_not_adapt_on_active_frames():
    tracker = BaselineTracker(calibration_seconds=1.0, min_samples=10, adapt_tau=5.0)
    for index in range(60):
        tracker.observe({"x": 1.0}, index / FPS)
    for index in range(60, 300):
        tracker.observe({"x": 2.0}, index / FPS, quiet=False)
    assert tracker.median("x") == pytest.approx(1.0, abs=0.01)


def test_reset_clears_the_baseline():
    tracker = BaselineTracker(calibration_seconds=1.0, min_samples=10)
    for index in range(60):
        tracker.observe({"x": 1.0}, index / FPS)
    assert tracker.ready
    tracker.reset()
    assert not tracker.ready
    assert tracker.progress == 0.0


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
def test_estimator_reports_every_stress_au():
    estimator = ActionUnitEstimator(calibration_seconds=1.0)
    frame = calibrate(estimator, seconds=2.0)
    assert set(frame.activations) == set(AU_ORDER) == set(STRESS_ACTION_UNITS)
    assert frame.calibrated


def test_neutral_face_scores_no_action_units():
    estimator = ActionUnitEstimator(calibration_seconds=1.0)
    frame = calibrate(estimator, seconds=3.0)
    for code, activation in frame.activations.items():
        assert activation.intensity < 1.0, f"{code} fired on a neutral face"


@pytest.mark.parametrize(
    ("code", "face_kwargs"),
    [
        ("AU1", {"brow_raise": 9.0}),                      # inner brow up
        ("AU4", {"brow_height": 6.0, "brow_separation": 14.0}),  # brow down + together
        ("AU6", {"cheek_height": 20.0}),                   # cheek up towards the lid
        ("AU7", {"lower_lid": 2.0}),                       # lower lid up
        ("AU12", {"corner_elevation": 7.0, "mouth_width": 56.0}),
        ("AU17", {"chin_distance": 20.0}),                 # chin boss up
        ("AU20", {"mouth_width": 62.0}),                   # stretch without elevation
        ("AU23", {"lip_thickness": 2.0}),                  # lips thinned
        ("AU24", {"lip_thickness": 2.5, "mouth_opening": 0.0}),
    ],
)
def test_geometric_rules_fire_the_right_action_unit(code, face_kwargs):
    estimator = ActionUnitEstimator(calibration_seconds=1.0)
    calibrate(estimator, seconds=3.0)

    features = geometric_features(make_face(**face_kwargs))
    frame = estimator.update(features, NO_MOTION, 3.5)
    assert frame.intensity(code) >= 1.0, f"{code} did not fire: {frame.activations[code]}"


def test_intensities_stay_within_the_facs_scale():
    estimator = ActionUnitEstimator(calibration_seconds=1.0)
    calibrate(estimator, seconds=3.0)
    extreme = geometric_features(
        make_face(brow_raise=30.0, mouth_width=90.0, lip_thickness=0.5, eye_aperture=0.5)
    )
    frame = estimator.update(extreme, NO_MOTION, 3.5)
    for activation in frame.activations.values():
        assert 0.0 <= activation.intensity <= 5.0


def test_au7_is_not_claimed_by_a_cheek_raise():
    """AU6 and AU7 both narrow the eye; the cheek term must keep them apart."""
    estimator = ActionUnitEstimator(calibration_seconds=1.0)
    calibrate(estimator, seconds=3.0)
    frame = estimator.update(geometric_features(make_face(cheek_height=18.0)), NO_MOTION, 3.5)
    assert frame.intensity("AU6") > frame.intensity("AU7")


def test_masked_smile_requires_au12_without_au6():
    estimator = ActionUnitEstimator(calibration_seconds=1.0)
    calibrate(estimator, seconds=3.0)

    smile_only = geometric_features(make_face(corner_elevation=8.0, mouth_width=58.0))
    assert estimator.update(smile_only, NO_MOTION, 3.5).masked_smile

    duchenne = geometric_features(
        make_face(corner_elevation=8.0, mouth_width=58.0, cheek_height=18.0)
    )
    assert not estimator.update(duchenne, NO_MOTION, 3.6).masked_smile


# --------------------------------------------------------------------------- #
# Motion channels
# --------------------------------------------------------------------------- #
def test_motion_channels_use_canonical_crop_signs():
    inward = {
        "brow_inner_right": RegionMotion("brow_inner_right", 0.8, -0.5, 1.0, 1.0, 1.0),
        "brow_inner_left": RegionMotion("brow_inner_left", -0.8, -0.5, 1.0, 1.0, 1.0),
    }
    channels = ActionUnitEstimator.motion_channels(inward)
    assert channels["brow_converge"] > 0      # brows drawn together (AU4)
    assert channels["brow_rise"] > 0          # both moving up

    apart = {
        "mouth_corner_right": RegionMotion("mouth_corner_right", -0.9, 0.0, 1.0, 1.0, 1.0),
        "mouth_corner_left": RegionMotion("mouth_corner_left", 0.9, 0.0, 1.0, 1.0, 1.0),
    }
    assert ActionUnitEstimator.motion_channels(apart)["mouth_spread"] > 0

    press = {
        "lip_upper": RegionMotion("lip_upper", 0.0, 0.6, 0.6, 1.0, 0.36),
        "lip_lower": RegionMotion("lip_lower", 0.0, -0.6, 0.6, 1.0, 0.36),
    }
    assert ActionUnitEstimator.motion_channels(press)["lip_press"] > 0


def test_missing_regions_default_to_no_motion():
    channels = ActionUnitEstimator.motion_channels({})
    assert set(channels)
    assert all(value == 0.0 for value in channels.values())


def test_motion_evidence_raises_intensity():
    """The same geometry plus coherent upward brow motion must score higher."""
    regions = STRESS_ACTION_UNITS["AU1"].regions
    quiet = ActionUnitEstimator(calibration_seconds=1.0)
    loud = ActionUnitEstimator(calibration_seconds=1.0)
    for estimator in (quiet, loud):
        calibrate(estimator, seconds=3.0)

    # A small brow raise: enough to register, far from saturation, so the
    # extra motion evidence still has room to show up.
    features = geometric_features(make_face(brow_raise=1.0))
    for index in range(20):  # build the motion channel history
        t = 3.5 + index / FPS
        quiet.update(features, motions_for(regions, dy=0.0), t)
        loud.update(features, motions_for(regions, dy=-0.01), t)

    t = 4.5
    quiet_frame = quiet.update(features, motions_for(regions, dy=0.0), t)
    loud_frame = loud.update(features, motions_for(regions, dy=-1.5), t)
    assert loud_frame.intensity("AU1") > quiet_frame.intensity("AU1")


# --------------------------------------------------------------------------- #
# Blinks and quality
# --------------------------------------------------------------------------- #
def test_blinks_are_counted_and_long_closures_are_not_blinks():
    estimator = ActionUnitEstimator(calibration_seconds=1.0)
    calibrate(estimator, seconds=3.0)

    t = 3.0
    for _ in range(4):
        for _ in range(3):  # ~100 ms closed
            estimator.update(geometric_features(make_face(eye_aperture=0.5, lower_lid=0.5)), NO_MOTION, t)
            t += 1 / FPS
        for _ in range(15):
            estimator.update(geometric_features(make_face()), NO_MOTION, t)
            t += 1 / FPS
    assert estimator._blinks and len(estimator._blinks) == 4

    for _ in range(40):  # a long closure is AU43, not a blink
        estimator.update(geometric_features(make_face(eye_aperture=0.5, lower_lid=0.5)), NO_MOTION, t)
        t += 1 / FPS
    frame = estimator.update(geometric_features(make_face()), NO_MOTION, t)
    assert len(estimator._blinks) == 4
    assert frame.blink_rate > 0.0


def test_eye_closure_drives_au43():
    estimator = ActionUnitEstimator(calibration_seconds=1.0)
    calibrate(estimator, seconds=3.0)
    closed = geometric_features(make_face(eye_aperture=0.5, lower_lid=0.5))
    frame = estimator.update(closed, NO_MOTION, 3.5)
    assert frame.eyes_closed
    assert frame.intensity("AU43") > 2.0


def test_quality_penalises_extreme_pose_and_tiny_faces():
    frontal = ActionUnitEstimator.frame_quality(
        {"yaw_ratio": 0.0, "roll_degrees": 0.0, "interocular_px": 90.0}
    )
    turned = ActionUnitEstimator.frame_quality(
        {"yaw_ratio": 0.55, "roll_degrees": 30.0, "interocular_px": 90.0}
    )
    tiny = ActionUnitEstimator.frame_quality(
        {"yaw_ratio": 0.0, "roll_degrees": 0.0, "interocular_px": 25.0}
    )
    assert frontal == pytest.approx(1.0)
    assert 0.0 <= turned < 0.3
    assert tiny == 0.0


def test_low_quality_frames_suppress_intensities():
    estimator = ActionUnitEstimator(calibration_seconds=1.0)
    calibrate(estimator, seconds=3.0)
    features = geometric_features(make_face(brow_raise=12.0))
    good = estimator.update(features, NO_MOTION, 3.5)

    poor = dict(features)
    poor["interocular_px"] = 32.0  # far away -> too few pixels for flow
    poor_frame = estimator.update(poor, NO_MOTION, 3.6)
    assert poor_frame.quality < good.quality
    assert poor_frame.intensity("AU1") < good.intensity("AU1")
