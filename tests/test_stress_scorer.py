"""Stress index: weighting, modifiers, smoothing and reporting."""

from __future__ import annotations

import numpy as np
import pytest

from src.au_estimator import AUActivation, AUFrame, STRESS_ACTION_UNITS
from src.stress_scorer import (
    AU_STRESS_WEIGHTS,
    StressScorer,
    stress_level,
)


def au_frame(intensities=None, timestamp=0.0, calibrated=True, **kwargs) -> AUFrame:
    """Build an :class:`AUFrame` with the given AU intensities."""
    intensities = intensities or {}
    activations = {
        code: AUActivation(code, spec.name, float(intensities.get(code, 0.0)), 0.0, 0.0)
        for code, spec in STRESS_ACTION_UNITS.items()
    }
    return AUFrame(timestamp=timestamp, activations=activations, calibrated=calibrated, **kwargs)


def settle(scorer: StressScorer, frame_kwargs, seconds=20.0, fps=30.0):
    """Run the smoother to convergence and return the final state."""
    state = None
    for index in range(int(seconds * fps)):
        state = scorer.update(au_frame(timestamp=index / fps, **frame_kwargs))
    return state


# --------------------------------------------------------------------------- #
# Levels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("score", "label"),
    [(0.0, "calm"), (1.9, "calm"), (2.0, "mild"), (3.9, "mild"),
     (4.0, "moderate"), (6.0, "elevated"), (8.0, "high"), (10.0, "high")],
)
def test_stress_level_bands(score, label):
    assert stress_level(score) == label


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_uncalibrated_frames_are_flagged_not_scored():
    scorer = StressScorer()
    state = scorer.update(au_frame({"AU4": 5.0}, calibrated=False))
    assert state.level == "calibrating"
    assert state.score == 0.0
    assert not state.calibrated


def test_neutral_face_scores_zero():
    state = settle(StressScorer(), {"intensities": {}})
    assert state.score == pytest.approx(0.0, abs=1e-6)
    assert state.level == "calm"


def test_score_is_monotonic_in_au_intensity():
    scores = []
    for intensity in (0.0, 1.0, 2.5, 4.0, 5.0):
        state = settle(StressScorer(), {"intensities": {"AU4": intensity}})
        scores.append(state.score)
    assert all(b >= a for a, b in zip(scores, scores[1:]))
    assert scores[-1] > scores[0]


def test_all_stress_aus_saturate_below_the_ceiling():
    maxed = {code: 5.0 for code in AU_STRESS_WEIGHTS if AU_STRESS_WEIGHTS[code] > 0}
    state = settle(StressScorer(), {"intensities": maxed})
    assert 0.0 <= state.score <= 10.0
    assert state.score > 8.0


def test_weights_order_the_action_units():
    """AU4 carries more weight than AU43, so it must move the index more."""
    au4 = settle(StressScorer(), {"intensities": {"AU4": 5.0}}).score
    au43 = settle(StressScorer(), {"intensities": {"AU43": 5.0}}).score
    assert au4 > au43


def test_duchenne_smile_lowers_the_score():
    stressed = {"AU4": 3.0, "AU7": 3.0}
    without = settle(StressScorer(), {"intensities": stressed}).score
    with_au6 = settle(StressScorer(), {"intensities": {**stressed, "AU6": 4.0}}).score
    assert with_au6 < without


def test_masked_smile_adds_a_penalty():
    plain = settle(StressScorer(), {"intensities": {"AU4": 2.0}})
    masked = settle(StressScorer(), {"intensities": {"AU4": 2.0, "AU12": 3.0}})
    assert masked.masked_smile
    assert not plain.masked_smile
    assert masked.components["masked_smile"] == pytest.approx(1.0)


def test_micro_expression_rate_contributes():
    quiet = settle(StressScorer(), {"intensities": {"AU4": 1.0}, "micro_rate": 0.0})
    busy = settle(StressScorer(), {"intensities": {"AU4": 1.0}, "micro_rate": 30.0})
    assert busy.score > quiet.score
    assert busy.components["micro_rate"] == pytest.approx(1.5)


def test_blink_rate_only_counts_above_the_neutral_range():
    normal = settle(StressScorer(), {"intensities": {}, "blink_rate": 15.0})
    high = settle(StressScorer(), {"intensities": {}, "blink_rate": 60.0})
    assert normal.components["blink_rate"] == 0.0
    assert high.components["blink_rate"] == pytest.approx(1.0)
    assert high.score > normal.score


def test_contributions_explain_the_score():
    state = settle(StressScorer(), {"intensities": {"AU4": 4.0, "AU24": 3.0}})
    assert state.contributions["AU4"] > state.contributions["AU24"] > 0
    drivers = [code for code, _ in state.top_drivers]
    assert drivers[:2] == ["AU4", "AU24"]
    assert sum(state.contributions.values()) == pytest.approx(state.components["au_load"])


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #
def test_smoothing_lags_a_step_change():
    scorer = StressScorer(smoothing_tau=2.0)
    scorer.update(au_frame({}, timestamp=0.0))
    stepped = scorer.update(au_frame({"AU4": 5.0}, timestamp=0.1))
    assert stepped.instant > stepped.score  # the smoothed index has not caught up
    assert stepped.score > 0.0


def test_smoothing_converges_to_the_instant_value():
    scorer = StressScorer(smoothing_tau=0.5)
    state = settle(scorer, {"intensities": {"AU4": 5.0}}, seconds=10.0)
    assert state.score == pytest.approx(state.instant, abs=0.05)


def test_score_is_clipped_to_the_scale():
    scorer = StressScorer(micro_weight=50.0)
    state = settle(scorer, {"intensities": {"AU4": 5.0}, "micro_rate": 100.0})
    assert state.score <= 10.0
    assert state.instant <= 10.0


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_history_series_and_summary():
    scorer = StressScorer(smoothing_tau=0.5)
    # A calm stretch followed by a stressed one, so mean and peak differ.
    for index in range(150):
        scorer.update(au_frame({"AU4": 1.0}, timestamp=index / 30.0))
    for index in range(150, 240):
        scorer.update(au_frame({"AU4": 5.0, "AU24": 5.0}, timestamp=index / 30.0))
    times, scores = scorer.series()
    assert times.size == scores.size > 0
    assert np.all(np.diff(times) >= 0)

    summary = scorer.summary()
    assert summary["frames"] > 0
    assert 0.0 < summary["mean_score"] < summary["peak_score"] <= 10.0
    assert summary["duration_s"] > 0


def test_summary_without_calibrated_frames():
    scorer = StressScorer()
    scorer.update(au_frame({}, calibrated=False))
    assert scorer.summary() == {"frames": 0}


def test_history_is_pruned_to_the_window():
    scorer = StressScorer(history_seconds=1.0)
    settle(scorer, {"intensities": {}}, seconds=5.0, fps=30.0)
    times, _ = scorer.series()
    assert times[-1] - times[0] <= 1.05


def test_to_row_is_csv_ready():
    state = settle(StressScorer(), {"intensities": {"AU4": 2.0}})
    row = state.to_row()
    assert row["level"] == state.level
    assert row["stress_score"] == pytest.approx(state.score, abs=1e-3)
    assert row["masked_smile"] in (0, 1)
    assert "contrib_AU4" in row
    assert all(not isinstance(value, (list, dict)) for value in row.values())


def test_reset_clears_the_smoother():
    scorer = StressScorer()
    settle(scorer, {"intensities": {"AU4": 5.0}}, seconds=5.0)
    scorer.reset()
    assert not scorer.history
    first = scorer.update(au_frame({}, timestamp=100.0))
    assert first.score == pytest.approx(0.0, abs=1e-6)


def test_calibrating_and_scored_rows_share_a_schema():
    """CSV headers are fixed by the first row, so the schema must not change."""
    scorer = StressScorer()
    calibrating = scorer.update(au_frame({"AU4": 5.0}, calibrated=False)).to_row()
    scored = scorer.update(au_frame({"AU4": 5.0}, timestamp=1.0)).to_row()
    assert set(calibrating) == set(scored)
    assert calibrating["contrib_AU4"] == 0.0
