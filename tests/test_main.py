"""CLI surface: argument parsing, CSV logging and console formatting."""

from __future__ import annotations

import csv

import pytest

import main as cli
from src.au_estimator import AUActivation, AUFrame, STRESS_ACTION_UNITS
from src.pipeline import PipelineResult
from src.stress_scorer import StressScorer


def au_frame(intensities=None, calibrated=True, **kwargs) -> AUFrame:
    intensities = intensities or {}
    activations = {
        code: AUActivation(code, spec.name, float(intensities.get(code, 0.0)), 0.0, 0.0)
        for code, spec in STRESS_ACTION_UNITS.items()
    }
    return AUFrame(timestamp=1.0, activations=activations, calibrated=calibrated, **kwargs)


def result_with(intensities=None, calibrated=True) -> PipelineResult:
    frame = au_frame(intensities, calibrated=calibrated)
    state = StressScorer(smoothing_tau=0.01).update(frame)
    return PipelineResult(timestamp=1.0, au_frame=frame, stress=state)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def test_defaults():
    args = cli.build_parser().parse_args([])
    assert args.camera == "0"
    assert args.show is True
    assert args.heatmap is True
    assert args.calibration == pytest.approx(3.0)


def test_camera_accepts_both_source_kinds():
    parser = cli.build_parser()
    assert parser.parse_args(["--camera", "0"]).camera == "0"
    url = "rtsp://admin:@192.168.1.15:554/stream1"
    assert parser.parse_args(["--camera", url]).camera == url


def test_no_show_is_headless():
    assert cli.build_parser().parse_args(["--no-show"]).show is False


def test_show_and_no_show_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--show", "--no-show"])


def test_output_flags():
    args = cli.build_parser().parse_args(
        ["--save", "out.mp4", "--csv", "log.csv", "--no-heatmap", "--mesh", "--max-frames", "10"]
    )
    assert args.save == "out.mp4"
    assert args.csv == "log.csv"
    assert args.heatmap is False
    assert args.mesh is True
    assert args.max_frames == 10


def test_unknown_log_level_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--log-level", "TRACE"])


# --------------------------------------------------------------------------- #
# CSV logging
# --------------------------------------------------------------------------- #
def test_csv_logger_writes_header_and_rows(tmp_path):
    path = tmp_path / "nested" / "log.csv"
    logger = cli.CsvLogger(str(path))
    logger.write(result_with({"AU4": 3.0}))
    logger.write(result_with({"AU4": 1.0}))
    logger.close()

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {"timestamp", "stress_score", "level", "AU4", "contrib_AU4"} <= set(rows[0])
    assert float(rows[0]["AU4"]) == pytest.approx(3.0)


def test_csv_logger_skips_unscored_frames(tmp_path):
    path = tmp_path / "log.csv"
    logger = cli.CsvLogger(str(path))
    logger.write(PipelineResult(timestamp=0.0))  # no face
    logger.close()
    assert path.read_text(encoding="utf-8") == ""


# --------------------------------------------------------------------------- #
# Console output
# --------------------------------------------------------------------------- #
def test_fps_meter():
    meter = cli.FpsMeter()
    assert meter.tick(0.0) == 0.0
    for index in range(1, 11):
        fps = meter.tick(index / 25.0)
    assert fps == pytest.approx(25.0, rel=1e-6)


def test_status_line_variants():
    assert "no face" in cli.format_status(PipelineResult(timestamp=0.0), 30.0)
    assert "calibrating" in cli.format_status(result_with(calibrated=False), 30.0)

    line = cli.format_status(result_with({"AU4": 4.0, "AU12": 3.0}), 29.5)
    assert "stress" in line
    assert "AU4" in line
    assert "MASKED SMILE" in line  # AU12 without AU6


def test_summary_prints_the_disclaimer(capsys):
    from src.pipeline import StressPipeline

    pipeline = StressPipeline.__new__(StressPipeline)  # no camera/MediaPipe needed
    pipeline.scorer = StressScorer(smoothing_tau=0.01)
    for index in range(30):
        pipeline.scorer.update(au_frame({"AU4": 2.0}))
    cli.print_summary(pipeline)

    output = capsys.readouterr().out
    assert "mean stress index" in output
    assert "not a diagnosis" in output


def test_summary_without_frames(capsys):
    from src.pipeline import StressPipeline

    pipeline = StressPipeline.__new__(StressPipeline)
    pipeline.scorer = StressScorer()
    cli.print_summary(pipeline)
    assert "No calibrated frames" in capsys.readouterr().out


def test_run_reports_a_bad_source():
    args = cli.build_parser().parse_args(["--camera", "/definitely/not/a/camera.mp4", "--no-show"])
    assert cli.run(args) == 2
