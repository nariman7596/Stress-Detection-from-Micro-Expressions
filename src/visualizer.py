"""Real-time overlay: stress gauge, AU bars and per-region heat map.

The overlay is drawn *onto* the camera frame (never appended beside it) so the
output size stays equal to the capture size -- that keeps ``--save`` writing a
well-formed video and makes the preview usable as a demo GIF.

DE: Echtzeit-Overlay -- Stress-Balken, AU-Intensitaeten, Regionen-Heatmap.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .au_estimator import AU_ORDER, AUFrame, STRESS_ACTION_UNITS
from .face_mesh import FACE_REGIONS, FaceLandmarks
from .stress_scorer import StressState

BGR = Tuple[int, int, int]

WHITE: BGR = (245, 245, 245)
GREY: BGR = (140, 140, 140)
DARK: BGR = (28, 28, 30)
ACCENT: BGR = (255, 190, 90)
ALERT: BGR = (70, 70, 235)

#: Stress colour ramp (BGR), from calm green through amber to red.
_STRESS_RAMP: Sequence[Tuple[float, BGR]] = (
    (0.0, (110, 200, 90)),
    (0.4, (80, 220, 220)),
    (0.7, (60, 150, 245)),
    (1.0, (60, 60, 240)),
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def ramp_color(fraction: float) -> BGR:
    """Interpolate the stress colour ramp at ``fraction`` in ``[0, 1]``."""
    fraction = float(np.clip(fraction, 0.0, 1.0))
    for (low, low_color), (high, high_color) in zip(_STRESS_RAMP, _STRESS_RAMP[1:]):
        if fraction <= high:
            span = max(high - low, 1e-6)
            t = (fraction - low) / span
            return tuple(int(round(a + (b - a) * t)) for a, b in zip(low_color, high_color))  # type: ignore[return-value]
    return _STRESS_RAMP[-1][1]


def _blend_rect(image: np.ndarray, box: Tuple[int, int, int, int], color: BGR, alpha: float) -> None:
    """Alpha-blend a filled rectangle in place."""
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = image[y0:y1, x0:x1]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def _blend_polygon(image: np.ndarray, points: np.ndarray, color: BGR, alpha: float) -> None:
    """Alpha-blend a filled convex polygon in place, working on its bbox only."""
    hull = cv2.convexHull(points.astype(np.int32))
    x, y, w, h = cv2.boundingRect(hull)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image.shape[1], x + w), min(image.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = image[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.fillConvexPoly(overlay, hull - np.array([[x0, y0]], dtype=np.int32), color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def region_intensities(au_frame: AUFrame) -> Dict[str, float]:
    """Strongest AU intensity touching each face region (drives the heat map)."""
    result: Dict[str, float] = {}
    for code, spec in STRESS_ACTION_UNITS.items():
        intensity = au_frame.intensity(code)
        for region in spec.regions:
            result[region] = max(result.get(region, 0.0), intensity)
    return result


class Visualizer:
    """Draw the full real-time overlay for one frame.

    Parameters
    ----------
    show_mesh:
        Draw the raw landmark cloud.  Useful while debugging alignment, noisy on
        a demo recording.
    show_heatmap:
        Tint each AU region by the intensity of the strongest AU that uses it.
    panel_alpha:
        Opacity of the side panel background.
    """

    def __init__(
        self,
        show_mesh: bool = False,
        show_heatmap: bool = True,
        show_panel: bool = True,
        panel_width: int = 250,
        panel_alpha: float = 0.72,
        au_codes: Iterable[str] = AU_ORDER,
    ) -> None:
        self.show_mesh = show_mesh
        self.show_heatmap = show_heatmap
        self.show_panel = show_panel
        self.panel_width = int(panel_width)
        self.panel_alpha = float(panel_alpha)
        self.au_codes = list(au_codes)
        self._event_log: List[str] = []

    # -------------------------------------------------------------- elements
    def _draw_heatmap(self, image: np.ndarray, landmarks: FaceLandmarks, au_frame: AUFrame) -> None:
        for region, intensity in region_intensities(au_frame).items():
            if intensity < 0.5 or region not in FACE_REGIONS:
                continue
            fraction = float(np.clip(intensity / 5.0, 0.0, 1.0))
            _blend_polygon(
                image,
                landmarks.region_points(region),
                ramp_color(fraction),
                alpha=0.15 + 0.45 * fraction,
            )

    def _draw_mesh(self, image: np.ndarray, landmarks: FaceLandmarks) -> None:
        for x, y in landmarks.xy().astype(np.int32):
            cv2.circle(image, (int(x), int(y)), 1, (90, 200, 120), -1, lineType=cv2.LINE_AA)

    def _draw_gauge(self, image: np.ndarray, state: StressState, origin: Tuple[int, int], width: int) -> int:
        """Horizontal 0-10 stress bar.  Returns the y coordinate below it."""
        x, y = origin
        height = 18
        fraction = float(np.clip(state.score / 10.0, 0.0, 1.0))
        color = ramp_color(fraction)

        cv2.rectangle(image, (x, y), (x + width, y + height), (60, 60, 62), -1)
        cv2.rectangle(image, (x, y), (x + int(width * fraction), y + height), color, -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (95, 95, 98), 1)
        for tick in range(1, 10):
            tx = x + int(width * tick / 10.0)
            cv2.line(image, (tx, y + height - 5), (tx, y + height), (95, 95, 98), 1)

        label = "calibrating..." if not state.calibrated else f"{state.score:4.1f}/10  {state.level.upper()}"
        cv2.putText(image, label, (x, y + height + 17), FONT, 0.52, color if state.calibrated else GREY, 1, cv2.LINE_AA)
        return y + height + 28

    def _draw_au_bars(self, image: np.ndarray, au_frame: AUFrame, origin: Tuple[int, int], width: int) -> int:
        x, y = origin
        cv2.putText(image, "ACTION UNITS", (x, y), FONT, 0.42, GREY, 1, cv2.LINE_AA)
        y += 12
        bar_x = x + 46
        # Leave room on the right for the micro-expression dot marker.
        bar_width = width - 46 - 12
        for code in self.au_codes:
            activation = au_frame.activations.get(code)
            intensity = activation.intensity if activation else 0.0
            fraction = float(np.clip(intensity / 5.0, 0.0, 1.0))
            color = ramp_color(fraction) if fraction > 0 else (70, 70, 72)
            cv2.putText(image, code, (x, y + 9), FONT, 0.42, WHITE if fraction > 0.2 else GREY, 1, cv2.LINE_AA)
            cv2.rectangle(image, (bar_x, y), (bar_x + bar_width, y + 10), (55, 55, 58), -1)
            cv2.rectangle(image, (bar_x, y), (bar_x + int(bar_width * fraction), y + 10), color, -1)
            if activation is not None and activation.micro:
                # A micro-expression burst fired in this AU's region this instant.
                cv2.circle(image, (bar_x + bar_width + 7, y + 5), 3, ACCENT, -1, lineType=cv2.LINE_AA)
            y += 15
        return y

    @staticmethod
    def _fit(text: str, max_width: int, scale: float) -> str:
        """Truncate ``text`` with an ellipsis so it fits ``max_width`` pixels."""
        if cv2.getTextSize(text, FONT, scale, 1)[0][0] <= max_width:
            return text
        while text and cv2.getTextSize(text + "...", FONT, scale, 1)[0][0] > max_width:
            text = text[:-1]
        return text + "..."

    def _panel_height(
        self,
        au_frame: Optional[AUFrame],
        state: Optional[StressState],
        stats: Sequence[str],
        extra: Sequence[str],
    ) -> int:
        """Height the panel needs, so its background matches its content."""
        height = 30 + 12                                   # header
        if state is not None:
            height += 46                                   # gauge + label
        if au_frame is not None:
            height += 16 + 15 * len(self.au_codes) + 6     # AU bar block
            height += 14 * len(stats)
            if au_frame.masked_smile:
                height += 34
        if self._event_log:
            height += 18 + 13 * min(len(self._event_log), 5)
        height += 14 * (len(extra) + 1) + 10               # footer + fps
        return height

    def _draw_panel(
        self,
        image: np.ndarray,
        au_frame: Optional[AUFrame],
        state: Optional[StressState],
        fps: float,
        extra: Sequence[str],
    ) -> None:
        height, width = image.shape[:2]
        panel_width = min(self.panel_width, width - 10)
        x = width - panel_width - 8

        stats: Sequence[str] = ()
        if au_frame is not None:
            stats = [
                f"micro-expr {au_frame.micro_rate:5.1f}/min",
                f"blink rate {au_frame.blink_rate:5.1f}/min",
                f"quality    {au_frame.quality:5.2f}",
            ]
            if not au_frame.calibrated:
                stats = [f"calibration {au_frame.calibration_progress * 100:4.0f}%", *stats]

        panel_height = min(self._panel_height(au_frame, state, stats, extra), height - 16)
        _blend_rect(image, (x - 8, 8, width - 8, 8 + panel_height), DARK, self.panel_alpha)

        y = 30
        cv2.putText(image, "STRESS INDEX", (x, y), FONT, 0.48, WHITE, 1, cv2.LINE_AA)
        y += 12

        if state is not None:
            y = self._draw_gauge(image, state, (x, y), panel_width)
        if au_frame is not None:
            y = self._draw_au_bars(image, au_frame, (x, y + 4), panel_width) + 6
            for line in stats:
                cv2.putText(image, line, (x, y), FONT, 0.42, GREY, 1, cv2.LINE_AA)
                y += 14

        if au_frame is not None and au_frame.masked_smile:
            y += 4
            cv2.putText(image, "MASKED SMILE", (x, y), FONT, 0.48, ALERT, 1, cv2.LINE_AA)
            cv2.putText(image, "AU12 without AU6", (x, y + 13), FONT, 0.38, ALERT, 1, cv2.LINE_AA)
            y += 30

        if self._event_log:
            y += 4
            cv2.putText(image, "RECENT LEAKS", (x, y), FONT, 0.42, GREY, 1, cv2.LINE_AA)
            y += 14
            for line in self._event_log[-5:]:
                cv2.putText(image, line, (x, y), FONT, 0.38, ACCENT, 1, cv2.LINE_AA)
                y += 13

        y += 6
        for line in [f"{fps:4.1f} fps", *extra]:
            cv2.putText(image, self._fit(line, panel_width, 0.4), (x, y), FONT, 0.4, GREY, 1, cv2.LINE_AA)
            y += 14

    # ----------------------------------------------------------------- public
    def render(
        self,
        frame: np.ndarray,
        landmarks: Optional[FaceLandmarks] = None,
        au_frame: Optional[AUFrame] = None,
        state: Optional[StressState] = None,
        fps: float = 0.0,
        extra_lines: Sequence[str] = (),
        copy: bool = True,
    ) -> np.ndarray:
        """Return ``frame`` with the full overlay drawn on it."""
        image = frame.copy() if copy else frame

        if landmarks is not None and au_frame is not None:
            if self.show_heatmap:
                self._draw_heatmap(image, landmarks, au_frame)
            if self.show_mesh:
                self._draw_mesh(image, landmarks)
            x0, y0, x1, y1 = landmarks.bbox
            cv2.rectangle(image, (x0, y0), (x1, y1), (120, 120, 125), 1)
            if au_frame.micro_events:
                # Every event in one frame belongs to the same leak; name the
                # strongest region and how many regions it moved.
                strongest = max(au_frame.micro_events, key=lambda e: e.peak_z)
                spread = len(au_frame.micro_events)
                self._event_log.append(
                    f"{strongest.region[:15]:<15} {strongest.duration * 1000:3.0f}ms x{spread}"
                )
                self._event_log = self._event_log[-20:]
        elif landmarks is None:
            cv2.putText(image, "no face detected", (16, 32), FONT, 0.6, ALERT, 2, cv2.LINE_AA)

        if self.show_panel:
            self._draw_panel(image, au_frame, state, fps, extra_lines)
        return image


def make_writer(path: str, fps: float, size: Tuple[int, int], fourcc: str = "mp4v") -> cv2.VideoWriter:
    """Create a video writer for ``size = (width, height)``.

    ``mp4v`` is used because it is available in every stock OpenCV build,
    including the wheels installed by ``pip``.
    """
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), max(fps, 1.0), size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path!r}")
    return writer
