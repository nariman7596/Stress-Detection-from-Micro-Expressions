"""Camera abstraction: source parsing, credential masking, file playback."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.camera import (
    CameraOpenError,
    CameraStream,
    describe_source,
    is_stream_url,
    open_camera,
    parse_source,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        ("2", 2),
        (1, 1),
        (" 0 ", 0),
        ("rtsp://admin:@192.168.1.15:554/stream1", "rtsp://admin:@192.168.1.15:554/stream1"),
        ("clip.mp4", "clip.mp4"),
    ],
)
def test_parse_source(value, expected):
    assert parse_source(value) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (0, False),
        ("rtsp://192.168.1.15:554/stream1", True),
        ("RTSP://192.168.1.15:554/stream1", True),
        ("http://cam/live", True),
        ("clip.mp4", False),
    ],
)
def test_is_stream_url(source, expected):
    assert is_stream_url(source) is expected


def test_describe_source_masks_credentials():
    described = describe_source("rtsp://admin:secret@192.168.1.15:554/stream1")
    assert "secret" not in described
    assert "192.168.1.15:554/stream1" in described


def test_describe_source_usb():
    assert describe_source(0) == "USB device index 0"


@pytest.fixture
def sample_video(tmp_path):
    """Write a tiny synthetic clip so playback can be tested without a camera."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
    assert writer.isOpened()
    for index in range(10):
        frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_stream_reads_video_file(sample_video):
    with CameraStream(str(sample_video)) as stream:
        frames = list(stream.frames())
    assert len(frames) == 10
    assert frames[0].image.shape == (48, 64, 3)
    assert [f.index for f in frames] == list(range(10))
    assert frames[-1].timestamp >= frames[0].timestamp


def test_max_frames_limits_iteration(sample_video):
    with CameraStream(str(sample_video)) as stream:
        assert len(list(stream.frames(max_frames=3))) == 3


def test_flip_mirrors_image(sample_video, monkeypatch):
    with CameraStream(str(sample_video)) as plain:
        original = plain.read().image
    with CameraStream(str(sample_video), flip=True) as flipped:
        mirrored = flipped.read().image
    assert np.array_equal(mirrored, cv2.flip(original, 1))


def test_open_camera_raises_on_bad_source(tmp_path):
    missing = tmp_path / "does-not-exist.mp4"
    with pytest.raises(CameraOpenError):
        open_camera(str(missing))


def test_measured_fps_is_positive(sample_video):
    with CameraStream(str(sample_video)) as stream:
        list(stream.frames())
        assert stream.measured_fps > 0.0
