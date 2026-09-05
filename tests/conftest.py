"""Shared fixtures: a synthetic, fully controllable face.

The tests must run without a camera and without MediaPipe, so the landmark
array is built analytically.  Only the indices the code actually reads are
placed meaningfully; the rest are scattered over a face-shaped ellipse so that
bounding boxes and hulls stay sane.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_mesh import FaceLandmarks  # noqa: E402

NUM_LANDMARKS = 478
CENTER = np.array([320.0, 240.0])
INTEROCULAR = 60.0  # pixels between the outer eye corners


def make_face(
    brow_raise: float = 0.0,
    brow_separation: float = 24.0,
    brow_height: float = 14.0,
    eye_aperture: float = 6.0,
    lower_lid: float = 6.0,
    cheek_height: float = 30.0,
    mouth_width: float = 44.0,
    corner_elevation: float = 0.0,
    lip_thickness: float = 5.0,
    mouth_opening: float = 0.0,
    chin_distance: float = 30.0,
    yaw_offset: float = 0.0,
    scale: float = 1.0,
    seed: int = 7,
) -> FaceLandmarks:
    """Build a synthetic :class:`FaceLandmarks` with controllable FACS geometry.

    All arguments are in pixels at ``scale=1``; increasing ``brow_raise`` lifts
    the inner brows, ``eye_aperture`` opens the eyes, and so on.
    """
    rng = np.random.default_rng(seed)
    points = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)

    # Scatter every landmark over a face-shaped ellipse first...
    angles = rng.uniform(0.0, 2.0 * np.pi, NUM_LANDMARKS)
    radii = rng.uniform(0.2, 1.0, NUM_LANDMARKS)
    points[:, 0] = CENTER[0] + np.cos(angles) * radii * 70.0
    points[:, 1] = CENTER[1] + np.sin(angles) * radii * 95.0

    half = INTEROCULAR / 2.0
    eye_y = -10.0

    def place(index: int, x: float, y: float) -> None:
        points[index, 0] = CENTER[0] + (x + yaw_offset) * scale
        points[index, 1] = CENTER[1] + y * scale

    # --- eyes -------------------------------------------------------------
    place(33, -half, eye_y)          # right outer corner
    place(133, -half / 3.0, eye_y)   # right inner corner
    place(362, half / 3.0, eye_y)    # left inner corner
    place(263, half, eye_y)          # left outer corner
    for upper, lower, x in ((159, 145, -half * 0.66), (386, 374, half * 0.66)):
        place(upper, x, eye_y - eye_aperture / 2.0)
        place(lower, x, eye_y + lower_lid / 2.0)
    # EAR support points, slightly inside the lid extremes
    for idx, x, y in (
        (160, -half * 0.8, eye_y - eye_aperture / 2.4),
        (158, -half * 0.5, eye_y - eye_aperture / 2.4),
        (144, -half * 0.8, eye_y + lower_lid / 2.4),
        (153, -half * 0.5, eye_y + lower_lid / 2.4),
        (385, half * 0.5, eye_y - eye_aperture / 2.4),
        (387, half * 0.8, eye_y - eye_aperture / 2.4),
        (380, half * 0.5, eye_y + lower_lid / 2.4),
        (373, half * 0.8, eye_y + lower_lid / 2.4),
    ):
        place(idx, x, y)

    # --- brows ------------------------------------------------------------
    place(107, -brow_separation / 2.0, eye_y - brow_height - brow_raise)
    place(336, brow_separation / 2.0, eye_y - brow_height - brow_raise)
    place(105, -half * 0.7, eye_y - brow_height)
    place(334, half * 0.7, eye_y - brow_height)
    place(70, -half * 1.1, eye_y - brow_height * 0.8)
    place(300, half * 1.1, eye_y - brow_height * 0.8)

    # --- nose (alignment anchors) -----------------------------------------
    place(168, 0.0, eye_y - 2.0)
    place(6, 0.0, eye_y + 6.0)
    place(197, 0.0, eye_y + 12.0)
    place(1, 0.0, eye_y + 20.0)
    place(2, 0.0, eye_y + 24.0)

    # --- cheeks -----------------------------------------------------------
    place(205, -half * 0.85, eye_y + cheek_height)
    place(425, half * 0.85, eye_y + cheek_height)

    # --- mouth ------------------------------------------------------------
    mouth_y = eye_y + 55.0
    place(61, -mouth_width / 2.0, mouth_y - corner_elevation)
    place(291, mouth_width / 2.0, mouth_y - corner_elevation)
    place(0, 0.0, mouth_y - mouth_opening / 2.0 - lip_thickness)
    place(13, 0.0, mouth_y - mouth_opening / 2.0)
    place(14, 0.0, mouth_y + mouth_opening / 2.0)
    place(17, 0.0, mouth_y + mouth_opening / 2.0 + lip_thickness)

    # --- chin -------------------------------------------------------------
    place(199, 0.0, mouth_y + chin_distance * 0.7)
    place(152, 0.0, mouth_y + chin_distance)

    return FaceLandmarks(points=points, image_shape=(480, 640))


@pytest.fixture
def neutral_face() -> FaceLandmarks:
    return make_face()


@pytest.fixture
def texture_image() -> np.ndarray:
    """A high-frequency BGR texture -- optical flow needs something to track."""
    import cv2

    rng = np.random.default_rng(3)
    noise = (rng.random((320, 320)) * 255).astype(np.uint8)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.0)
    return cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)


def features_of(landmarks: FaceLandmarks) -> Dict[str, float]:
    from src.face_mesh import geometric_features

    return geometric_features(landmarks)
