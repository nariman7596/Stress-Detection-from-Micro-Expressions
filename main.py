#!/usr/bin/env python3
"""Real-time stress detection from facial micro-expressions.

Examples
--------
    # USB webcam (Creative Live! Cam), live preview
    python main.py --camera 0 --show

    # WiFi RTSP camera, no window, log the score to CSV
    python main.py --camera rtsp://admin:@192.168.1.15:554/stream1 --no-show --csv session.csv

    # Re-analyse a recording and write an annotated video
    python main.py --camera clip.mp4 --save annotated.mp4

Keys (with ``--show``): ``q``/ESC quit, ``m`` toggle mesh, ``h`` toggle heat map,
``r`` restart the neutral-baseline calibration.

NOT A MEDICAL DEVICE.  Research and portfolio use only -- see README.
"""

from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Optional, TextIO

import cv2

from src.camera import CameraOpenError, CameraStream, describe_source, parse_source
from src.pipeline import PipelineResult, StressPipeline
from src.visualizer import Visualizer, make_writer

logger = logging.getLogger("stress-detector")

WINDOW_NAME = "Micro-Expression Stress Detector"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Real-time stress estimation from facial micro-expressions (FACS Action Units).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Research tool. Not a medical device and not a lie detector.",
    )

    source = parser.add_argument_group("source")
    source.add_argument(
        "--camera",
        default="0",
        help="USB device index (e.g. 0), RTSP URL (rtsp://user:pass@host:554/stream1) or a video file path",
    )
    source.add_argument("--width", type=int, default=None, help="requested capture width")
    source.add_argument("--height", type=int, default=None, help="requested capture height")
    source.add_argument("--fps", type=float, default=None, help="requested capture frame rate")
    source.add_argument("--flip", action="store_true", help="mirror the image (natural for a webcam)")
    source.add_argument("--max-frames", type=int, default=None, help="stop after N frames")

    output = parser.add_argument_group("output")
    show = output.add_mutually_exclusive_group()
    show.add_argument("--show", dest="show", action="store_true", default=True, help="open a preview window")
    show.add_argument("--no-show", dest="show", action="store_false", help="headless mode")
    output.add_argument("--save", metavar="PATH", default=None, help="write the annotated video to PATH")
    output.add_argument("--csv", metavar="PATH", default=None, help="log per-frame scores to PATH")
    output.add_argument("--print-interval", type=float, default=2.0,
                        help="seconds between console score updates (0 to disable)")
    output.add_argument("--mesh", action="store_true", help="draw the raw landmark cloud")
    output.add_argument("--no-heatmap", dest="heatmap", action="store_false", default=True,
                        help="disable the AU region heat map")

    analysis = parser.add_argument_group("analysis")
    analysis.add_argument("--calibration", type=float, default=3.0,
                          help="seconds of neutral face used to build the personal baseline")
    analysis.add_argument("--align-size", type=int, default=256, help="canonical face crop size in pixels")
    analysis.add_argument("--flow-downscale", type=float, default=0.5,
                          help="optical flow resolution factor (lower = faster)")
    analysis.add_argument("--micro-sensitivity", type=float, default=1.0,
                          help="micro-expression sensitivity; >1 catches fainter leaks, <1 is stricter")
    analysis.add_argument("--smoothing", type=float, default=1.5,
                          help="stress index smoothing time constant in seconds")
    analysis.add_argument("--min-confidence", type=float, default=0.5,
                          help="MediaPipe detection/tracking confidence")
    analysis.add_argument("--model-asset", default=None,
                          help="path to face_landmarker.task (MediaPipe >= 1.0 backend only)")

    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="logging verbosity")
    return parser


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class CsvLogger:
    """Append per-frame stress rows to a CSV, writing the header on first use."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: Optional[TextIO] = self.path.open("w", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None

    def write(self, result: PipelineResult) -> None:
        if result.stress is None or self._handle is None:
            return
        row = result.stress.to_row()
        if result.au_frame is not None:
            for code, activation in result.au_frame.activations.items():
                row[code] = round(activation.intensity, 3)
        if self._writer is None:
            self._writer = csv.DictWriter(self._handle, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class FpsMeter:
    """Frame rate over a trailing window."""

    def __init__(self, window: int = 30) -> None:
        self._times: Deque[float] = deque(maxlen=window)

    def tick(self, timestamp: float) -> float:
        self._times.append(timestamp)
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


def format_status(result: PipelineResult, fps: float) -> str:
    """One-line console status for the current frame."""
    if result.stress is None or result.au_frame is None:
        return f"[{fps:4.1f} fps] no face"
    state, au = result.stress, result.au_frame
    if not state.calibrated:
        return f"[{fps:4.1f} fps] calibrating {au.calibration_progress * 100:3.0f}%"
    drivers = ", ".join(f"{code} {value:+.1f}" for code, value in state.top_drivers[:3]) or "-"
    flag = "  MASKED SMILE" if state.masked_smile else ""
    return (
        f"[{fps:4.1f} fps] stress {state.score:4.1f}/10 ({state.level:<8}) "
        f"micro {au.micro_rate:4.1f}/min  blink {au.blink_rate:4.1f}/min  "
        f"drivers: {drivers}{flag}"
    )


def print_summary(pipeline: StressPipeline) -> None:
    summary = pipeline.scorer.summary()
    if not summary.get("frames"):
        print("\nNo calibrated frames were scored.")
        return
    print("\n--- session summary ---")
    print(f"scored frames      : {int(summary['frames'])}")
    print(f"duration           : {summary['duration_s']:.1f} s")
    print(f"mean stress index  : {summary['mean_score']:.2f}/10")
    print(f"95th percentile    : {summary['p95_score']:.2f}/10")
    print(f"peak stress index  : {summary['peak_score']:.2f}/10")
    print(f"masked-smile frames: {int(summary['masked_smile_frames'])}")
    print(f"micro-expr rate    : {summary['mean_micro_rate']:.1f} /min")
    print(f"blink rate         : {summary['mean_blink_rate']:.1f} /min")
    print("This is a research read-out, not a diagnosis.")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    source = parse_source(args.camera)
    logger.info("Source: %s", describe_source(source))

    visualizer = Visualizer(show_mesh=args.mesh, show_heatmap=args.heatmap)
    fps_meter = FpsMeter()
    csv_logger = CsvLogger(args.csv) if args.csv else None
    writer: Optional[cv2.VideoWriter] = None
    last_print = 0.0
    stopping = {"flag": False}

    def handle_sigint(signum, frame):  # noqa: ARG001 - signal API
        stopping["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        stream = CameraStream(
            source,
            width=args.width,
            height=args.height,
            fps=args.fps,
            flip=args.flip,
        ).open()
    except CameraOpenError as exc:
        logger.error("%s", exc)
        return 2

    pipeline = StressPipeline(
        calibration_seconds=args.calibration,
        align_size=args.align_size,
        flow_downscale=args.flow_downscale,
        micro_sensitivity=args.micro_sensitivity,
        min_detection_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence,
        model_asset_path=args.model_asset,
        smoothing_tau=args.smoothing,
    )
    logger.info("MediaPipe backend: %s", pipeline.detector.backend)
    print("Hold a relaxed, neutral, forward-facing expression during calibration.")

    try:
        for frame in stream.frames(max_frames=args.max_frames):
            if stopping["flag"]:
                break

            result = pipeline.process(frame.image, frame.timestamp)
            fps = fps_meter.tick(frame.timestamp)

            extra = [f"backend {pipeline.detector.backend}", describe_source(source)]
            rendered = visualizer.render(
                frame.image,
                landmarks=result.landmarks,
                au_frame=result.au_frame,
                state=result.stress,
                fps=fps,
                extra_lines=extra,
            )

            if csv_logger is not None:
                csv_logger.write(result)

            if args.save:
                if writer is None:
                    height, width = rendered.shape[:2]
                    writer = make_writer(args.save, fps or stream.reported_fps or 25.0, (width, height))
                    logger.info("Recording annotated video to %s", args.save)
                writer.write(rendered)

            if args.print_interval > 0 and frame.timestamp - last_print >= args.print_interval:
                print(format_status(result, fps))
                last_print = frame.timestamp

            if args.show:
                cv2.imshow(WINDOW_NAME, rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    visualizer.show_mesh = not visualizer.show_mesh
                if key == ord("h"):
                    visualizer.show_heatmap = not visualizer.show_heatmap
                if key == ord("r"):
                    pipeline.estimator.baseline.reset()
                    pipeline.scorer.reset()
                    logger.info("Baseline calibration restarted")
    finally:
        stream.release()
        pipeline.close()
        if writer is not None:
            writer.release()
        if csv_logger is not None:
            csv_logger.close()
        if args.show:
            cv2.destroyAllWindows()

    print_summary(pipeline)
    if args.csv:
        print(f"Per-frame log written to {args.csv}")
    if args.save:
        print(f"Annotated video written to {args.save}")
    return 0


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
