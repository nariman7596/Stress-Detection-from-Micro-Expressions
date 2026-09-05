"""Camera-agnostic video capture.

The project must run against two very different sources:

* a **USB webcam** (Creative Live! Cam) opened through ``cv2.VideoCapture(0)``
  -- on macOS this goes through the AVFoundation backend;
* a **WiFi RTSP camera** (V380 fisheye, H.265) reached over the LAN, e.g.
  ``rtsp://admin:@192.168.1.15:554/stream1``.

Both are hidden behind :class:`CameraStream`, which also handles the two things
that bite in practice with network cameras: the FFmpeg backend defaulting to UDP
(packet loss -> smeared frames on H.265) and silent stream drops that need a
reconnect.

DE: Kameraquelle wird per ``--camera`` gewaehlt (Index ODER RTSP-URL).
"""

from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Source = Union[int, str]

#: URL schemes that we treat as "network stream" (needs FFmpeg + TCP transport).
_STREAM_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def parse_source(value: Union[int, str]) -> Source:
    """Turn a CLI ``--camera`` value into something ``cv2.VideoCapture`` accepts.

    ``"0"`` becomes the integer ``0`` (device index), everything else is kept as
    a string (RTSP URL or path to a video file).

    >>> parse_source("0")
    0
    >>> parse_source("rtsp://cam/stream1")
    'rtsp://cam/stream1'
    """
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def is_stream_url(source: Source) -> bool:
    """True if ``source`` is a network stream URL rather than a device/file."""
    return isinstance(source, str) and source.lower().startswith(_STREAM_SCHEMES)


def describe_source(source: Source) -> str:
    """Human readable source label with RTSP credentials masked."""
    if isinstance(source, int):
        return f"USB device index {source}"
    if is_stream_url(source) and "@" in source:
        scheme, _, rest = source.partition("://")
        _, _, host = rest.rpartition("@")
        return f"{scheme}://***@{host}"
    return str(source)


def _select_backend(source: Source) -> int:
    """Pick the most reliable OpenCV backend for this source and platform."""
    if is_stream_url(source):
        return cv2.CAP_FFMPEG
    if isinstance(source, int):
        system = platform.system()
        if system == "Darwin":
            return cv2.CAP_AVFOUNDATION
        if system == "Linux":
            return cv2.CAP_V4L2
        if system == "Windows":
            return cv2.CAP_DSHOW
    return cv2.CAP_ANY


@dataclass
class Frame:
    """A single captured frame plus the metadata the pipeline needs."""

    image: np.ndarray
    index: int
    timestamp: float  # seconds, monotonic clock

    @property
    def shape(self) -> tuple:
        return self.image.shape


class CameraOpenError(RuntimeError):
    """Raised when a capture source cannot be opened at all."""


class CameraStream:
    """Uniform capture wrapper over USB webcams, RTSP cameras and video files.

    Parameters
    ----------
    source:
        Device index (``int``), RTSP URL or path to a video file.
    width, height:
        Requested capture resolution.  Ignored by most RTSP cameras (the stream
        dictates its own size), honoured by USB cameras.
    fps:
        Requested capture frame rate (USB only, best effort).
    flip:
        Mirror the image horizontally -- natural for a webcam facing the user.
    reconnect_attempts, reconnect_delay:
        Network cameras drop out; on read failure the stream is reopened with an
        exponential backoff, up to this many times per failure episode.
    buffer_size:
        ``CAP_PROP_BUFFERSIZE``.  ``1`` keeps latency low, which matters because
        micro-expressions last 40-200 ms and we want to react live.
    """

    def __init__(
        self,
        source: Source,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        flip: bool = False,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
        buffer_size: int = 1,
        rtsp_transport: str = "tcp",
    ) -> None:
        self.source = parse_source(source)
        self.width = width
        self.height = height
        self.fps = fps
        self.flip = flip
        self.reconnect_attempts = max(0, reconnect_attempts)
        self.reconnect_delay = reconnect_delay
        self.buffer_size = buffer_size
        self.rtsp_transport = rtsp_transport

        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------ open
    def open(self) -> "CameraStream":
        """Open the capture device, raising :class:`CameraOpenError` on failure."""
        self._capture = self._create_capture()
        if not self._capture.isOpened():
            raise CameraOpenError(
                f"Could not open camera source: {describe_source(self.source)}. "
                "For a USB camera check the device index and macOS camera permission; "
                "for RTSP check URL, credentials and that the host is reachable."
            )
        self._start_time = time.monotonic()
        logger.info(
            "Camera opened: %s (%dx%d @ %.1f fps)",
            describe_source(self.source),
            self.frame_width,
            self.frame_height,
            self.reported_fps,
        )
        return self

    def _create_capture(self) -> cv2.VideoCapture:
        if is_stream_url(self.source):
            # Must be set *before* the VideoCapture is constructed: FFmpeg reads
            # it once at open time.  UDP + H.265 over WiFi = torn frames.
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                f"rtsp_transport;{self.rtsp_transport}|stimeout;5000000",
            )
        capture = cv2.VideoCapture(self.source, _select_backend(self.source))
        if self.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.fps:
            capture.set(cv2.CAP_PROP_FPS, float(self.fps))
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, float(self.buffer_size))
        except cv2.error:  # pragma: no cover - backend dependent
            pass
        return capture

    # ------------------------------------------------------------------ read
    def read(self) -> Optional[Frame]:
        """Return the next :class:`Frame`, or ``None`` when the stream ended.

        A failed read on a network stream triggers a reconnect; a failed read on
        a video file simply means end-of-file.
        """
        if self._capture is None:
            self.open()
        assert self._capture is not None

        ok, image = self._capture.read()
        if not ok or image is None:
            if not self._reconnect():
                return None
            assert self._capture is not None
            ok, image = self._capture.read()
            if not ok or image is None:
                return None

        if self.flip:
            image = cv2.flip(image, 1)

        frame = Frame(image=image, index=self._frame_index, timestamp=time.monotonic())
        self._frame_index += 1
        return frame

    def _reconnect(self) -> bool:
        """Try to reopen a dropped stream.  Only network sources are retried."""
        if not is_stream_url(self.source) or self.reconnect_attempts == 0:
            return False

        for attempt in range(1, self.reconnect_attempts + 1):
            delay = self.reconnect_delay * (2 ** (attempt - 1))
            logger.warning(
                "Stream read failed (%s); reconnect attempt %d/%d in %.1fs",
                describe_source(self.source),
                attempt,
                self.reconnect_attempts,
                delay,
            )
            time.sleep(delay)
            self.release()
            try:
                capture = self._create_capture()
            except cv2.error:  # pragma: no cover - backend dependent
                continue
            if capture.isOpened():
                self._capture = capture
                logger.info("Reconnected to %s", describe_source(self.source))
                return True
            capture.release()
        logger.error("Giving up on %s after %d attempts", describe_source(self.source), self.reconnect_attempts)
        return False

    def frames(self, max_frames: Optional[int] = None) -> Iterator[Frame]:
        """Iterate frames until the stream ends or ``max_frames`` are produced."""
        produced = 0
        while max_frames is None or produced < max_frames:
            frame = self.read()
            if frame is None:
                return
            produced += 1
            yield frame

    # ------------------------------------------------------------ properties
    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def frame_width(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._capture else 0

    @property
    def frame_height(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._capture else 0

    @property
    def reported_fps(self) -> float:
        """FPS as reported by the backend (0 or nonsense for many webcams)."""
        if self._capture is None:
            return 0.0
        value = float(self._capture.get(cv2.CAP_PROP_FPS))
        return value if 0.0 < value < 1000.0 else 0.0

    @property
    def measured_fps(self) -> float:
        """Actual delivery rate measured since :meth:`open`."""
        if self._start_time is None or self._frame_index == 0:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        return self._frame_index / elapsed if elapsed > 0 else 0.0

    # ----------------------------------------------------------- lifecycle
    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    close = release

    def __enter__(self) -> "CameraStream":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __iter__(self) -> Iterator[Frame]:
        return self.frames()


def open_camera(source: Source, **kwargs) -> CameraStream:
    """Convenience factory mirroring ``watcher.py``'s ``open_camera()``."""
    return CameraStream(source, **kwargs).open()
