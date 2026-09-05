"""Micro-motion extraction: flow accuracy, region aggregation, burst detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.optical_flow import (
    MICRO_MAX_DURATION,
    MicroExpressionDetector,
    MicroMotionExtractor,
    RegionMotion,
    TemporalSignal,
    aggregate_regions,
    band_energy_ratio,
    region_motion,
)


def shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_first_frame_has_no_flow(texture_image):
    extractor = MicroMotionExtractor()
    assert extractor.update(texture_image, 0.0) is None


def test_flow_recovers_a_known_translation(texture_image):
    extractor = MicroMotionExtractor(compensate_global=False)
    extractor.update(texture_image, 0.0)
    result = extractor.update(shift(texture_image, 2.0, -1.0), 1 / 30)

    assert result is not None
    motion = region_motion(result.flow, (80, 80, 240, 240), "patch")
    assert motion.dx == pytest.approx(2.0, abs=0.35)
    assert motion.dy == pytest.approx(-1.0, abs=0.35)
    assert motion.coherence > 0.9
    assert motion.rise == pytest.approx(1.0, abs=0.35)
    assert result.dt == pytest.approx(1 / 30, rel=1e-6)


def test_global_compensation_removes_rigid_motion(texture_image):
    """A whole-face translation is head motion, not expression: it must cancel."""
    extractor = MicroMotionExtractor(compensate_global=True)
    extractor.update(texture_image, 0.0)
    result = extractor.update(shift(texture_image, 3.0, 2.0), 1 / 30)

    motion = region_motion(result.flow, (80, 80, 240, 240), "patch")
    assert abs(motion.dx) < 0.3
    assert abs(motion.dy) < 0.3
    assert result.global_motion[0] == pytest.approx(3.0, abs=0.4)
    assert result.global_motion[1] == pytest.approx(2.0, abs=0.4)


def test_reset_drops_the_reference_frame(texture_image):
    extractor = MicroMotionExtractor()
    extractor.update(texture_image, 0.0)
    extractor.reset()
    assert extractor.update(texture_image, 0.1) is None


def test_downscale_is_rescaled_back_to_crop_pixels(texture_image):
    coarse = MicroMotionExtractor(downscale=0.5, compensate_global=False)
    coarse.update(texture_image, 0.0)
    result = coarse.update(shift(texture_image, 2.0, 0.0), 1 / 30)
    assert result.flow.shape[:2] == texture_image.shape[:2]
    assert region_motion(result.flow, (80, 80, 240, 240)).dx == pytest.approx(2.0, abs=0.5)


def test_invalid_downscale_rejected():
    with pytest.raises(ValueError):
        MicroMotionExtractor(downscale=0.0)


# --------------------------------------------------------------------------- #
# Region aggregation
# --------------------------------------------------------------------------- #
def test_region_motion_on_degenerate_box():
    flow = np.zeros((32, 32, 2), dtype=np.float32)
    motion = region_motion(flow, (10, 10, 11, 11), "tiny")
    assert motion == RegionMotion("tiny", 0.0, 0.0, 0.0, 0.0, 0.0)


def test_incoherent_noise_has_low_coherence():
    rng = np.random.default_rng(1)
    flow = rng.normal(scale=1.0, size=(64, 64, 2)).astype(np.float32)
    motion = region_motion(flow, (0, 0, 64, 64), "noise")
    assert motion.magnitude > 0.5
    assert motion.coherence < 0.15
    assert motion.directed < 0.2  # noise is suppressed by the coherence weighting


def test_aggregate_regions_covers_every_box():
    flow = np.ones((64, 64, 2), dtype=np.float32)
    boxes = {"a": (0, 0, 32, 32), "b": (32, 32, 64, 64)}
    motions = aggregate_regions(flow, boxes)
    assert set(motions) == {"a", "b"}
    assert all(m.dx == pytest.approx(1.0) for m in motions.values())


# --------------------------------------------------------------------------- #
# Temporal analysis
# --------------------------------------------------------------------------- #
def test_temporal_signal_drops_old_samples():
    signal = TemporalSignal(history_seconds=1.0)
    for i in range(30):
        signal.append(i * 0.1, float(i))
    assert len(signal) <= 11
    assert signal.times.min() >= 1.9


def test_robust_stats_ignore_a_single_outlier():
    signal = TemporalSignal(history_seconds=10.0)
    for i in range(40):
        signal.append(i * 0.1, 1.0 + 0.01 * (i % 2))
    signal.append(4.1, 50.0)
    median, sigma = signal.robust_stats()
    assert median == pytest.approx(1.0, abs=0.02)
    assert sigma < 1.0
    assert signal.zscore(50.0) > 10.0


def test_band_energy_ratio_separates_drift_from_bursts():
    times = np.linspace(0.0, 2.0, 120)
    slow = np.sin(2 * np.pi * 0.3 * times)
    fast = np.sin(2 * np.pi * 12.0 * times)
    assert band_energy_ratio(times, fast) > 0.8
    assert band_energy_ratio(times, slow) < 0.3


def test_band_energy_ratio_needs_enough_samples():
    assert band_energy_ratio(np.arange(3.0), np.ones(3)) == 0.0


# --------------------------------------------------------------------------- #
# Burst detection
# --------------------------------------------------------------------------- #
def _run_detector(detector, values, fps=60.0, region="brow_inner_right"):
    events = []
    for index, value in enumerate(values):
        timestamp = index / fps
        motion = RegionMotion(region, 0.0, 0.0, value, 1.0, value ** 2)
        events.extend(detector.update({region: motion}, timestamp))
    return events


def test_detects_a_micro_expression_burst():
    """A ~120 ms burst on a quiet baseline is exactly what we want to catch."""
    detector = MicroExpressionDetector(regions=["brow_inner_right"])
    rng = np.random.default_rng(5)
    values = list(0.05 + rng.normal(scale=0.005, size=60))
    values += [1.2] * 7          # ~117 ms at 60 fps
    values += list(0.05 + rng.normal(scale=0.005, size=60))

    events = _run_detector(detector, values)
    assert len(events) == 1
    event = events[0]
    assert event.region == "brow_inner_right"
    assert 0.04 <= event.duration <= MICRO_MAX_DURATION
    assert event.peak_z > 5.0
    assert event.onset <= event.apex <= event.offset


def test_sustained_expression_is_not_a_micro_expression():
    detector = MicroExpressionDetector(regions=["chin"])
    rng = np.random.default_rng(6)
    values = list(0.05 + rng.normal(scale=0.005, size=60))
    values += [1.2] * 120        # 2 s -> a macro expression
    values += list(0.05 + rng.normal(scale=0.005, size=60))

    assert _run_detector(detector, values, region="chin") == []


def test_quiet_signal_produces_no_events():
    detector = MicroExpressionDetector(regions=["lip_upper"])
    rng = np.random.default_rng(7)
    values = list(0.05 + rng.normal(scale=0.005, size=200))
    assert _run_detector(detector, values, region="lip_upper") == []


def test_event_rate_and_memory():
    detector = MicroExpressionDetector(regions=["cheek_left"], event_memory=1.0)
    rng = np.random.default_rng(8)
    values = list(0.05 + rng.normal(scale=0.005, size=60)) + [1.2] * 7
    values += list(0.05 + rng.normal(scale=0.005, size=30))
    _run_detector(detector, values, region="cheek_left")

    assert detector.event_rate(1.6, window=10.0) > 0.0
    detector._prune(20.0)  # events older than event_memory are forgotten
    assert detector.recent_events(20.0) == []


def test_events_across_regions_count_as_one_leak():
    """One physical micro-expression moves many regions; the rate must not multiply."""
    regions = ["brow_inner_right", "brow_inner_left", "cheek_right", "cheek_left"]
    detector = MicroExpressionDetector(regions=regions)
    rng = np.random.default_rng(9)

    for index in range(60):
        quiet = {r: RegionMotion(r, 0.0, 0.0, 0.05 + rng.normal(scale=0.005), 1.0, 0.0) for r in regions}
        detector.update(quiet, index / 60.0)
    for index in range(60, 67):
        burst = {r: RegionMotion(r, 0.0, 0.0, 1.2, 1.0, 1.44) for r in regions}
        detector.update(burst, index / 60.0)
    events = []
    for index in range(67, 120):
        quiet = {r: RegionMotion(r, 0.0, 0.0, 0.05 + rng.normal(scale=0.005), 1.0, 0.0) for r in regions}
        events.extend(detector.update(quiet, index / 60.0))

    assert len(events) == len(regions)                       # one per region...
    assert len(detector.clustered_events(events)) == 1        # ...but one leak
    assert detector.event_rate(2.0, window=10.0) == pytest.approx(30.0, rel=0.05)


def test_event_rate_normalises_by_the_time_observed():
    """A 2 s session with one leak is 30/min, not 6/min."""
    detector = MicroExpressionDetector(regions=["chin"])
    rng = np.random.default_rng(10)
    values = list(0.05 + rng.normal(scale=0.005, size=60)) + [1.2] * 7
    values += list(0.05 + rng.normal(scale=0.005, size=53))
    _run_detector(detector, values, region="chin")
    assert detector.event_rate(2.0, window=10.0) == pytest.approx(30.0, rel=0.05)


def test_motion_below_the_absolute_floor_never_fires():
    """A still subject has a near-zero MAD: relative criteria alone would fire."""
    detector = MicroExpressionDetector(regions=["chin"], min_motion=0.15)
    rng = np.random.default_rng(11)
    values = list(0.02 + rng.normal(scale=0.0005, size=60))
    values += [0.05] * 7      # a huge z-score, but a physically meaningless motion
    values += list(0.02 + rng.normal(scale=0.0005, size=60))
    assert _run_detector(detector, values, region="chin") == []
