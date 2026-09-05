"""Landmark geometry, canonical alignment and the FACS measurement layer."""

from __future__ import annotations

import numpy as np
import pytest

from src.face_mesh import (
    FACE_REGIONS,
    STABLE_ANCHORS,
    FaceAligner,
    FaceLandmarks,
    apply_transform,
    geometric_features,
    similarity_transform,
)

from conftest import make_face


def test_landmarks_reject_wrong_shape():
    with pytest.raises(ValueError):
        FaceLandmarks(points=np.zeros((10, 2)), image_shape=(48, 64))


def test_interocular_and_regions(neutral_face):
    assert neutral_face.interocular == pytest.approx(60.0, abs=1e-3)
    for region in FACE_REGIONS:
        points = neutral_face.region_points(region)
        assert points.shape[1] == 2
        x0, y0, x1, y1 = neutral_face.region_box(region)
        assert x1 > x0 and y1 > y0


def test_region_box_is_clipped_to_the_image(neutral_face):
    height, width = neutral_face.image_shape
    for region in FACE_REGIONS:
        x0, y0, x1, y1 = neutral_face.region_box(region, padding=5.0)
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height


def test_unknown_region_raises(neutral_face):
    with pytest.raises(KeyError):
        neutral_face.region_points("left_elbow")


# --------------------------------------------------------------------------- #
# Similarity transform
# --------------------------------------------------------------------------- #
def test_similarity_transform_recovers_known_pose():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(8, 2)) * 40.0
    angle = np.deg2rad(23.0)
    scale = 1.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    translation = np.array([12.0, -5.0])
    dst = src @ (scale * rotation).T + translation

    matrix = similarity_transform(src, dst)
    assert apply_transform(src, matrix) == pytest.approx(dst, abs=1e-6)
    recovered_scale = np.sqrt(abs(np.linalg.det(matrix[:, :2])))
    assert recovered_scale == pytest.approx(scale, rel=1e-6)


def test_similarity_transform_rejects_mismatched_input():
    with pytest.raises(ValueError):
        similarity_transform(np.zeros((3, 2)), np.zeros((4, 2)))


def test_similarity_transform_is_not_affine():
    """A sheared target cannot be matched exactly -- shear must not be absorbed."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sheared = src @ np.array([[1.0, 0.6], [0.0, 1.0]])
    residual = apply_transform(src, similarity_transform(src, sheared)) - sheared
    assert np.abs(residual).max() > 0.05


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def test_aligner_normalises_pose_and_scale(texture_image):
    aligner = FaceAligner(size=128)
    straight = make_face()
    aligned = aligner.align(texture_image, straight)

    assert aligned.image.shape == (128, 128, 3)
    right_eye = aligned.landmarks.points[33, :2]
    left_eye = aligned.landmarks.points[263, :2]
    # The anchors are fitted in the least-squares sense, so the eye corners land
    # near -- not exactly on -- the canonical positions.
    assert right_eye[0] == pytest.approx(0.26 * 128, abs=5.0)
    assert left_eye[0] == pytest.approx(0.74 * 128, abs=5.0)
    # The eye line must be horizontal and centred, which is what actually
    # removes in-plane rotation and translation.
    assert right_eye[1] == pytest.approx(left_eye[1], abs=1.0)
    assert (right_eye[0] + left_eye[0]) / 2.0 == pytest.approx(64.0, abs=1.0)


def test_alignment_is_invariant_to_camera_distance(texture_image):
    aligner = FaceAligner(size=128)
    near = aligner.align(texture_image, make_face(scale=1.0)).landmarks
    far = aligner.align(texture_image, make_face(scale=0.5)).landmarks
    # Same expression at half the size must land on the same canonical points.
    assert near.points[list(STABLE_ANCHORS), :2] == pytest.approx(
        far.points[list(STABLE_ANCHORS), :2], abs=1.0
    )


def test_to_original_round_trips(texture_image, neutral_face):
    aligner = FaceAligner(size=128)
    aligned = aligner.align(texture_image, neutral_face)
    crop_points = aligned.landmarks.points[[33, 263, 61], :2]
    back = aligned.to_original(crop_points)
    assert back == pytest.approx(neutral_face.points[[33, 263, 61], :2], abs=1e-3)


# --------------------------------------------------------------------------- #
# FACS features
# --------------------------------------------------------------------------- #
def test_features_are_scale_invariant():
    near = geometric_features(make_face(scale=1.0))
    far = geometric_features(make_face(scale=0.5))
    for name in ("mouth_width", "eye_aperture", "inner_brow_raise", "chin_lip_distance"):
        assert near[name] == pytest.approx(far[name], rel=1e-4), name
    assert far["interocular_px"] == pytest.approx(30.0, abs=1e-3)


@pytest.mark.parametrize(
    ("kwargs", "feature", "direction"),
    [
        ({"brow_raise": 8.0}, "inner_brow_raise", +1),          # AU1
        ({"brow_height": 8.0}, "brow_lid_distance", -1),        # AU4 (brow to lid)
        ({"brow_separation": 16.0}, "brow_separation", -1),     # AU4 (drawn together)
        ({"eye_aperture": 2.0}, "eye_aperture", -1),            # AU43 / AU7
        ({"lower_lid": 2.0}, "lower_lid_gap", -1),              # AU7
        ({"cheek_height": 22.0}, "cheek_lid_distance", -1),     # AU6
        ({"mouth_width": 60.0}, "mouth_width", +1),             # AU12 / AU20
        ({"corner_elevation": 6.0}, "corner_elevation", +1),    # AU12
        ({"lip_thickness": 2.0}, "lip_thickness", -1),          # AU23 / AU24
        ({"mouth_opening": 8.0}, "mouth_opening", +1),
        ({"chin_distance": 20.0}, "chin_lip_distance", -1),     # AU17
    ],
)
def test_feature_responds_in_the_expected_direction(kwargs, feature, direction):
    neutral = geometric_features(make_face())
    changed = geometric_features(make_face(**kwargs))
    delta = changed[feature] - neutral[feature]
    assert np.sign(delta) == direction, f"{feature} moved by {delta:+.4f}"


def test_pose_features():
    assert geometric_features(make_face())["yaw_ratio"] == pytest.approx(0.0, abs=1e-6)
    rolled = make_face()
    rolled.points[263, 1] += 20.0  # tilt the left eye downwards
    assert geometric_features(rolled)["roll_degrees"] > 5.0


def test_alignment_cancels_in_plane_rotation(texture_image):
    """A rotated head must produce the same canonical landmarks as a straight one."""
    aligner = FaceAligner(size=128)
    straight = make_face()

    angle = np.deg2rad(18.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = straight.xy().mean(axis=0)
    rotated_points = straight.points.copy()
    rotated_points[:, :2] = (straight.xy() - center) @ rotation.T + center
    rotated = FaceLandmarks(rotated_points, straight.image_shape)

    a = aligner.align(texture_image, straight).landmarks.points[:, :2]
    b = aligner.align(texture_image, rotated).landmarks.points[:, :2]
    assert a == pytest.approx(b, abs=0.5)
