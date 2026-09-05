"""Overlay rendering: geometry preservation, heat map mapping, video writing."""

from __future__ import annotations

import numpy as np
import pytest

from src.au_estimator import AUActivation, AUFrame, STRESS_ACTION_UNITS
from src.stress_scorer import StressScorer
from src.visualizer import Visualizer, make_writer, ramp_color, region_intensities

from conftest import make_face


def build_au_frame(intensities=None, **kwargs) -> AUFrame:
    intensities = intensities or {}
    activations = {
        code: AUActivation(code, spec.name, float(intensities.get(code, 0.0)), 0.0, 0.0)
        for code, spec in STRESS_ACTION_UNITS.items()
    }
    return AUFrame(timestamp=1.0, activations=activations, calibrated=True, **kwargs)


@pytest.fixture
def frame() -> np.ndarray:
    return np.full((480, 640, 3), 40, dtype=np.uint8)


def test_ramp_is_continuous_and_bounded():
    for fraction in np.linspace(0.0, 1.0, 25):
        color = ramp_color(float(fraction))
        assert len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)
    assert ramp_color(-5.0) == ramp_color(0.0)
    assert ramp_color(99.0) == ramp_color(1.0)


def test_region_intensities_take_the_strongest_au():
    au = build_au_frame({"AU1": 4.0, "AU4": 2.0})  # both use the inner brow regions
    intensities = region_intensities(au)
    assert intensities["brow_inner_right"] == 4.0
    assert intensities["cheek_left"] == 0.0


def test_render_preserves_frame_geometry(frame):
    rendered = Visualizer().render(frame, fps=30.0)
    assert rendered.shape == frame.shape
    assert rendered.dtype == frame.dtype


def test_render_does_not_mutate_the_input(frame):
    original = frame.copy()
    Visualizer().render(frame, fps=30.0)
    assert np.array_equal(frame, original)


def test_render_in_place_when_copy_disabled(frame):
    rendered = Visualizer().render(frame, fps=30.0, copy=False)
    assert rendered is frame


def test_render_marks_a_missing_face(frame):
    rendered = Visualizer().render(frame, landmarks=None, fps=12.0)
    assert not np.array_equal(rendered, frame)  # the warning was drawn


def test_render_with_full_state(frame):
    scorer = StressScorer()
    state = scorer.update(build_au_frame({"AU4": 4.0, "AU12": 3.0}))
    rendered = Visualizer(show_mesh=True).render(
        frame,
        landmarks=make_face(),
        au_frame=build_au_frame({"AU4": 4.0, "AU12": 3.0}),
        state=state,
        fps=27.5,
        extra_lines=["backend solutions"],
    )
    assert rendered.shape == frame.shape
    assert rendered.max() > frame.max()  # something bright was drawn


def test_heatmap_can_be_disabled(frame):
    landmarks = make_face()
    au = build_au_frame({"AU4": 5.0})
    with_heat = Visualizer(show_heatmap=True, show_panel=False).render(frame, landmarks, au)
    without = Visualizer(show_heatmap=False, show_panel=False).render(frame, landmarks, au)
    assert not np.array_equal(with_heat, without)


def test_masked_smile_banner_changes_the_panel(frame):
    plain = build_au_frame({"AU12": 0.0})
    masked = build_au_frame({"AU12": 3.0})  # AU12 without AU6
    assert masked.masked_smile and not plain.masked_smile
    visualizer = Visualizer()
    assert not np.array_equal(
        visualizer.render(frame, make_face(), plain),
        Visualizer().render(frame, make_face(), masked),
    )


def test_render_survives_landmarks_outside_the_frame():
    small = np.zeros((120, 160, 3), dtype=np.uint8)
    landmarks = make_face()  # positioned for a 640x480 frame
    rendered = Visualizer().render(small, landmarks, build_au_frame({"AU4": 5.0}))
    assert rendered.shape == small.shape


def test_make_writer_round_trips(tmp_path):
    import cv2

    path = tmp_path / "out.mp4"
    writer = make_writer(str(path), 20.0, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    ok, image = capture.read()
    capture.release()
    assert ok and image.shape == (48, 64, 3)


def test_make_writer_rejects_a_bad_path(tmp_path):
    with pytest.raises(RuntimeError):
        make_writer(str(tmp_path / "nope" / "deep" / "out.mp4"), 20.0, (64, 48))


def test_panel_shrinks_to_its_content(frame):
    """The backdrop must track the content, not swallow a tall portrait frame."""
    visualizer = Visualizer()
    bare = visualizer._panel_height(None, None, (), ())
    full = visualizer._panel_height(
        build_au_frame({"AU12": 4.0}), StressScorer().update(build_au_frame()), ("a", "b", "c"), ("x",)
    )
    assert 0 < bare < full


def test_long_footer_lines_are_truncated(frame):
    long_line = "rtsp://" + "a" * 200 + "/stream1"
    rendered = Visualizer().render(frame, fps=30.0, extra_lines=[long_line])
    assert rendered.shape == frame.shape
    assert Visualizer._fit(long_line, 200, 0.4).endswith("...")
    assert Visualizer._fit("short", 200, 0.4) == "short"


def test_panel_never_exceeds_the_frame():
    tiny = np.zeros((90, 200, 3), dtype=np.uint8)
    rendered = Visualizer().render(tiny, make_face(), build_au_frame({"AU4": 5.0}), fps=10.0)
    assert rendered.shape == tiny.shape
