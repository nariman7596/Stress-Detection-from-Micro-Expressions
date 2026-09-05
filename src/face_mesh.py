"""MediaPipe FaceMesh wrapper, canonical face alignment and landmark geometry.

Three responsibilities live here:

1. :class:`FaceMeshDetector` -- a thin, backend-agnostic wrapper around MediaPipe.
   It supports both the legacy ``mediapipe.solutions.face_mesh`` API (MediaPipe
   0.10.x) and the newer Tasks API ``FaceLandmarker`` (MediaPipe >= 1.0), so the
   project keeps working across the 0.10 -> 1.0 break.
2. :class:`FaceAligner` -- a similarity (rotation + uniform scale + translation)
   warp into a canonical square crop.  Micro-expressions are *sub-millimetre*
   motions; without cancelling head pose and distance the optical flow field is
   dominated by rigid head motion.  A similarity transform is used deliberately:
   it removes rigid motion while preserving the non-rigid facial deformation we
   are trying to measure.
3. :func:`geometric_features` -- FACS-relevant distances/ratios computed from the
   landmarks, all normalised by the inter-ocular distance so they are invariant
   to camera distance.  This is a pure function of a landmark array, which keeps
   it unit-testable without a camera or MediaPipe installed.

DE: Landmark-Extraktion + Gesichts-Normalisierung + FACS-Messgroessen.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Landmark index tables (MediaPipe FaceMesh, 468 points; 478 with iris refine)
#
# Naming follows the *subject's* anatomy: "left" is the subject's left, which
# appears on the right-hand side of a non-mirrored image.
# --------------------------------------------------------------------------- #

# Single reference points
NOSE_TIP = 1
NOSE_BASE = 2          # subnasale -- rigid, good alignment anchor
NOSE_BRIDGE = 168      # between the eyes -- rigid
FOREHEAD = 10
CHIN_BOTTOM = 152
GLABELLA = 9

EYE_RIGHT_OUTER, EYE_RIGHT_INNER = 33, 133
EYE_LEFT_INNER, EYE_LEFT_OUTER = 362, 263
LID_RIGHT_UPPER, LID_RIGHT_LOWER = 159, 145
LID_LEFT_UPPER, LID_LEFT_LOWER = 386, 374

BROW_RIGHT_INNER, BROW_RIGHT_MID, BROW_RIGHT_OUTER = 107, 105, 70
BROW_LEFT_INNER, BROW_LEFT_MID, BROW_LEFT_OUTER = 336, 334, 300

CHEEK_RIGHT, CHEEK_LEFT = 205, 425
INFRAORBITAL_RIGHT, INFRAORBITAL_LEFT = 119, 348

MOUTH_CORNER_RIGHT, MOUTH_CORNER_LEFT = 61, 291
LIP_UPPER_OUTER, LIP_UPPER_INNER = 0, 13
LIP_LOWER_INNER, LIP_LOWER_OUTER = 14, 17
CHIN_BOSS = 199

#: Six-point sets used for the eye aspect ratio (EAR).
EAR_RIGHT = (33, 160, 158, 133, 153, 144)
EAR_LEFT = (362, 385, 387, 263, 373, 380)

#: Landmarks that barely move with facial expression -- used to estimate the
#: rigid head transform.  Eye *outer* corners, nose bridge and nose base.
STABLE_ANCHORS: Tuple[int, ...] = (33, 263, 168, 2, 6, 197)

#: AU-relevant regions of interest.  Optical flow is aggregated inside the
#: bounding box of each of these point sets.
FACE_REGIONS: Dict[str, Tuple[int, ...]] = {
    "brow_inner_right": (107, 55, 65, 66, 9),
    "brow_inner_left": (336, 285, 295, 296, 9),
    "brow_outer_right": (70, 63, 105, 46, 53),
    "brow_outer_left": (300, 293, 334, 276, 283),
    "eye_right": EAR_RIGHT + (157, 154, 173, 246),
    "eye_left": EAR_LEFT + (384, 381, 398, 466),
    "lid_lower_right": (145, 144, 153, 163, 7, 22, 23, 24),
    "lid_lower_left": (374, 373, 380, 390, 249, 252, 253, 254),
    "cheek_right": (205, 50, 101, 118, 119, 117, 123, 116),
    "cheek_left": (425, 280, 330, 347, 348, 346, 352, 345),
    "mouth_corner_right": (61, 146, 91, 185, 40, 76),
    "mouth_corner_left": (291, 375, 321, 409, 270, 306),
    "lip_upper": (0, 37, 39, 267, 269, 13, 82, 312),
    "lip_lower": (17, 84, 314, 14, 87, 317, 181, 405),
    "chin": (199, 175, 152, 18, 83, 313, 200),
    "nose": (1, 2, 4, 5, 6, 168, 197, 195),
}

_TASKS_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


# --------------------------------------------------------------------------- #
# Landmark container
# --------------------------------------------------------------------------- #
@dataclass
class FaceLandmarks:
    """Landmarks in *pixel* coordinates plus a few cached derived quantities.

    ``points`` has shape ``(N, 3)``: ``x`` and ``y`` in pixels, ``z`` in the same
    scale as ``x`` (MediaPipe's relative depth; only used for a coarse pose hint).
    """

    points: np.ndarray
    image_shape: Tuple[int, int]  # (height, width)
    tracked: bool = True

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {self.points.shape}")

    # ------------------------------------------------------------------ views
    def xy(self) -> np.ndarray:
        """``(N, 2)`` view of the pixel coordinates."""
        return self.points[:, :2]

    def region_points(self, region: str) -> np.ndarray:
        """``(k, 2)`` pixel coordinates of a named region."""
        try:
            indices = FACE_REGIONS[region]
        except KeyError as exc:  # pragma: no cover - programming error
            raise KeyError(f"unknown region '{region}'") from exc
        return self.points[list(indices), :2]

    def region_box(self, region: str, padding: float = 0.15) -> Tuple[int, int, int, int]:
        """Axis-aligned bounding box ``(x0, y0, x1, y1)`` around a region.

        ``padding`` is relative to the box size and expands it, because the AU
        motion of interest often peaks just outside the landmark ring (e.g. the
        infraorbital fold for AU6).
        """
        pts = self.region_points(region)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pad_x = max((x1 - x0) * padding, 2.0)
        pad_y = max((y1 - y0) * padding, 2.0)
        height, width = self.image_shape
        return (
            int(max(0, np.floor(x0 - pad_x))),
            int(max(0, np.floor(y0 - pad_y))),
            int(min(width, np.ceil(x1 + pad_x))),
            int(min(height, np.ceil(y1 + pad_y))),
        )

    # ------------------------------------------------------------- geometry
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        xy = self.xy()
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        return int(x0), int(y0), int(x1), int(y1)

    @property
    def interocular(self) -> float:
        """Outer-corner eye distance in pixels -- the normalisation scale."""
        return float(np.linalg.norm(self.points[EYE_LEFT_OUTER, :2] - self.points[EYE_RIGHT_OUTER, :2]))

    @property
    def eye_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        right = self.points[[EYE_RIGHT_OUTER, EYE_RIGHT_INNER], :2].mean(axis=0)
        left = self.points[[EYE_LEFT_OUTER, EYE_LEFT_INNER], :2].mean(axis=0)
        return right, left

    @property
    def roll_degrees(self) -> float:
        """In-plane head rotation, positive = head tilted to the subject's left."""
        right, left = self.eye_centers
        delta = left - right
        return float(np.degrees(np.arctan2(delta[1], delta[0])))

    @property
    def yaw_ratio(self) -> float:
        """Crude yaw estimate in ``[-1, 1]``; 0 = facing the camera.

        Ratio of the nose-tip offset from the eye midpoint to half the
        inter-ocular distance.  Used only to gate low-quality frames.
        """
        right, left = self.eye_centers
        mid = (right + left) / 2.0
        nose = self.points[NOSE_TIP, :2]
        half = max(self.interocular / 2.0, 1e-6)
        return float(np.clip((nose[0] - mid[0]) / half, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# Canonical alignment
# --------------------------------------------------------------------------- #
def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama) mapping ``src`` onto ``dst``.

    Returns a ``2x3`` matrix ``M`` such that ``dst ~= src @ M[:, :2].T + M[:, 2]``.
    Rotation + uniform scale + translation only -- no shear, no anisotropic
    scaling, so facial deformation survives the warp.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src and dst must both have shape (n, 2)")
    n = src.shape[0]
    if n < 2:
        raise ValueError("need at least two point correspondences")

    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    covariance = dst_c.T @ src_c / n

    u, singular_values, vt = np.linalg.svd(covariance)
    signs = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        signs[-1] = -1.0
    rotation = u @ np.diag(signs) @ vt

    src_var = (src_c ** 2).sum() / n
    scale = float((singular_values * signs).sum() / src_var) if src_var > 1e-12 else 1.0

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * rotation @ src_mean
    return matrix


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a ``2x3`` affine matrix to an ``(n, 2)`` array of points."""
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:, :2].T + matrix[:, 2]


@dataclass
class AlignedFace:
    """A canonically aligned face crop and everything needed to map back."""

    image: np.ndarray           # (size, size, 3) BGR
    landmarks: FaceLandmarks    # landmarks in crop coordinates
    matrix: np.ndarray          # 2x3 original -> crop
    size: int

    def to_original(self, points: np.ndarray) -> np.ndarray:
        """Map crop-space points back into original-image coordinates."""
        full = np.vstack([self.matrix, [0.0, 0.0, 1.0]])
        return apply_transform(points, np.linalg.inv(full)[:2])


class FaceAligner:
    """Warp faces into a fixed canonical frame so flow measures *expression*.

    The canonical layout puts the outer eye corners on a horizontal line and the
    nose base at a fixed height, which fixes scale, in-plane rotation and
    translation.  Out-of-plane rotation is *not* corrected -- large yaw/pitch is
    instead flagged as low quality by the AU estimator.
    """

    def __init__(self, size: int = 256) -> None:
        self.size = int(size)
        s = float(self.size)
        # Canonical destinations for STABLE_ANCHORS (fractions of the crop).
        self._canonical = np.array(
            [
                (0.26, 0.40),   # 33  right eye outer corner
                (0.74, 0.40),   # 263 left eye outer corner
                (0.50, 0.36),   # 168 nose bridge
                (0.50, 0.66),   # 2   nose base
                (0.50, 0.44),   # 6   nose upper
                (0.50, 0.58),   # 197 nose middle
            ],
            dtype=np.float64,
        ) * s

    def align(self, image: np.ndarray, landmarks: FaceLandmarks) -> AlignedFace:
        """Return the canonical crop of ``image`` for the given landmarks."""
        src = landmarks.points[list(STABLE_ANCHORS), :2].astype(np.float64)
        matrix = similarity_transform(src, self._canonical)
        crop = cv2.warpAffine(
            image,
            matrix.astype(np.float32),
            (self.size, self.size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        warped = apply_transform(landmarks.points[:, :2], matrix)
        scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
        points = np.column_stack([warped, landmarks.points[:, 2] * scale]).astype(np.float32)
        return AlignedFace(
            image=crop,
            landmarks=FaceLandmarks(points, (self.size, self.size), tracked=landmarks.tracked),
            matrix=matrix,
            size=self.size,
        )


# --------------------------------------------------------------------------- #
# FACS-relevant geometric features
# --------------------------------------------------------------------------- #
def _dist(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a, :2] - points[b, :2]))


def _eye_aspect_ratio(points: np.ndarray, indices: Sequence[int]) -> float:
    """Soukupova & Cech eye aspect ratio: (|p2-p6| + |p3-p5|) / (2 |p1-p4|)."""
    p1, p2, p3, p4, p5, p6 = (points[i, :2] for i in indices)
    horizontal = float(np.linalg.norm(p1 - p4))
    if horizontal < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal))


def geometric_features(landmarks: FaceLandmarks) -> Dict[str, float]:
    """Compute the scale-invariant FACS measurements used by the AU estimator.

    Every distance is divided by the inter-ocular distance, so the values are
    dimensionless and comparable across subjects and camera distances.  Signs are
    chosen so that *larger = more of the named action* wherever possible.
    """
    p = landmarks.points
    iod = max(landmarks.interocular, 1e-6)

    # --- brows ------------------------------------------------------------
    inner_brow_raise = (
        (p[EYE_RIGHT_INNER, 1] - p[BROW_RIGHT_INNER, 1]) + (p[EYE_LEFT_INNER, 1] - p[BROW_LEFT_INNER, 1])
    ) / (2.0 * iod)
    outer_brow_raise = (
        (p[EYE_RIGHT_OUTER, 1] - p[BROW_RIGHT_OUTER, 1]) + (p[EYE_LEFT_OUTER, 1] - p[BROW_LEFT_OUTER, 1])
    ) / (2.0 * iod)
    brow_lid_distance = (
        (p[LID_RIGHT_UPPER, 1] - p[BROW_RIGHT_MID, 1]) + (p[LID_LEFT_UPPER, 1] - p[BROW_LEFT_MID, 1])
    ) / (2.0 * iod)
    brow_separation = _dist(p, BROW_RIGHT_INNER, BROW_LEFT_INNER) / iod

    # --- eyes -------------------------------------------------------------
    ear_right = _eye_aspect_ratio(p, EAR_RIGHT)
    ear_left = _eye_aspect_ratio(p, EAR_LEFT)
    eye_aperture = (ear_right + ear_left) / 2.0
    # Split the aperture into the two lids: AU7 raises the lower lid, AU43/45
    # lowers the upper lid.  Both shrink the EAR, only the split tells them apart.
    right_axis = (p[EYE_RIGHT_OUTER, 1] + p[EYE_RIGHT_INNER, 1]) / 2.0
    left_axis = (p[EYE_LEFT_OUTER, 1] + p[EYE_LEFT_INNER, 1]) / 2.0
    upper_lid_gap = (
        (right_axis - p[LID_RIGHT_UPPER, 1]) + (left_axis - p[LID_LEFT_UPPER, 1])
    ) / (2.0 * iod)
    lower_lid_gap = (
        (p[LID_RIGHT_LOWER, 1] - right_axis) + (p[LID_LEFT_LOWER, 1] - left_axis)
    ) / (2.0 * iod)

    # --- cheeks (AU6) -----------------------------------------------------
    cheek_lid_distance = (
        (p[CHEEK_RIGHT, 1] - p[LID_RIGHT_LOWER, 1]) + (p[CHEEK_LEFT, 1] - p[LID_LEFT_LOWER, 1])
    ) / (2.0 * iod)

    # --- mouth ------------------------------------------------------------
    mouth_width = _dist(p, MOUTH_CORNER_RIGHT, MOUTH_CORNER_LEFT) / iod
    mouth_opening = _dist(p, LIP_UPPER_INNER, LIP_LOWER_INNER) / iod
    lip_center_y = (p[LIP_UPPER_OUTER, 1] + p[LIP_LOWER_OUTER, 1]) / 2.0
    corner_elevation = (
        (lip_center_y - p[MOUTH_CORNER_RIGHT, 1]) + (lip_center_y - p[MOUTH_CORNER_LEFT, 1])
    ) / (2.0 * iod)
    upper_lip_thickness = _dist(p, LIP_UPPER_OUTER, LIP_UPPER_INNER) / iod
    lower_lip_thickness = _dist(p, LIP_LOWER_INNER, LIP_LOWER_OUTER) / iod
    lip_thickness = upper_lip_thickness + lower_lip_thickness

    # --- chin (AU17) ------------------------------------------------------
    chin_lip_distance = (p[CHIN_BOTTOM, 1] - p[LIP_LOWER_OUTER, 1]) / iod
    chin_boss_distance = (p[CHIN_BOSS, 1] - p[LIP_LOWER_OUTER, 1]) / iod

    return {
        "inner_brow_raise": float(inner_brow_raise),
        "outer_brow_raise": float(outer_brow_raise),
        "brow_lid_distance": float(brow_lid_distance),
        "brow_separation": float(brow_separation),
        "eye_aperture": float(eye_aperture),
        "eye_aperture_right": float(ear_right),
        "eye_aperture_left": float(ear_left),
        "upper_lid_gap": float(upper_lid_gap),
        "lower_lid_gap": float(lower_lid_gap),
        "cheek_lid_distance": float(cheek_lid_distance),
        "mouth_width": float(mouth_width),
        "mouth_opening": float(mouth_opening),
        "corner_elevation": float(corner_elevation),
        "upper_lip_thickness": float(upper_lip_thickness),
        "lower_lip_thickness": float(lower_lip_thickness),
        "lip_thickness": float(lip_thickness),
        "chin_lip_distance": float(chin_lip_distance),
        "chin_boss_distance": float(chin_boss_distance),
        "interocular_px": float(iod),
        "roll_degrees": float(landmarks.roll_degrees),
        "yaw_ratio": float(landmarks.yaw_ratio),
    }


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
def ensure_tasks_model(path: Optional[Path] = None, url: str = _TASKS_MODEL_URL) -> Path:
    """Return a local ``face_landmarker.task`` bundle, downloading it if needed.

    Only used by the MediaPipe >= 1.0 (Tasks) backend; the 0.10 solutions API
    ships its model inside the wheel.
    """
    target = Path(path) if path else _DEFAULT_MODEL_DIR / "face_landmarker.task"
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading FaceLandmarker model to %s", target)
    tmp = target.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed, vendor-controlled URL
    os.replace(tmp, target)
    return target


class FaceMeshDetector:
    """Detect 468/478 facial landmarks, on whichever MediaPipe generation is installed.

    Parameters
    ----------
    static_image_mode:
        ``False`` (default) enables MediaPipe's internal tracking, which is both
        faster and temporally smoother -- important because we differentiate the
        landmark signal over time.
    refine_landmarks:
        Adds the iris/eyelid refinement points (478 landmarks total).  Improves
        the eyelid measurements that AU7 and AU43/45 depend on.
    model_asset_path:
        Optional path to ``face_landmarker.task`` for the Tasks backend.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_asset_path: Optional[str] = None,
    ) -> None:
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        self.model_asset_path = model_asset_path

        self.backend: str = "none"
        self._impl = None
        self._init_backend()

    # ------------------------------------------------------------- backends
    def _init_backend(self) -> None:
        try:
            import mediapipe as mp  # noqa: PLC0415 - optional heavy import
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "mediapipe is required for face landmark detection. "
                "Install it with `pip install -r requirements.txt`."
            ) from exc

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.backend = "solutions"
            self._impl = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=self.static_image_mode,
                max_num_faces=self.max_num_faces,
                refine_landmarks=self.refine_landmarks,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            logger.info("FaceMesh backend: mediapipe.solutions (v%s)", mp.__version__)
            return

        from mediapipe.tasks import python as mp_python  # noqa: PLC0415
        from mediapipe.tasks.python import vision  # noqa: PLC0415

        model_path = ensure_tasks_model(Path(self.model_asset_path) if self.model_asset_path else None)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE if self.static_image_mode else vision.RunningMode.VIDEO,
            num_faces=self.max_num_faces,
            min_face_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.backend = "tasks"
        self._mp = mp
        self._impl = vision.FaceLandmarker.create_from_options(options)
        logger.info("FaceMesh backend: mediapipe.tasks FaceLandmarker (v%s)", mp.__version__)

    # ---------------------------------------------------------------- process
    def process(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> Optional[FaceLandmarks]:
        """Detect the primary face in a BGR frame.

        Returns ``None`` when no face is found.  ``timestamp_ms`` is required by
        the Tasks video backend and must be monotonically increasing.
        """
        if self._impl is None:  # pragma: no cover - defensive
            raise RuntimeError("detector is closed")

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "solutions":
            result = self._impl.process(rgb)
            faces = getattr(result, "multi_face_landmarks", None)
            if not faces:
                return None
            normalized = np.array([[lm.x, lm.y, lm.z] for lm in faces[0].landmark], dtype=np.float32)
        else:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            if self.static_image_mode:
                result = self._impl.detect(image)
            else:
                result = self._impl.detect_for_video(image, int(timestamp_ms if timestamp_ms is not None else 0))
            if not result.face_landmarks:
                return None
            normalized = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]], dtype=np.float32
            )

        points = normalized.copy()
        points[:, 0] *= width
        points[:, 1] *= height
        points[:, 2] *= width  # MediaPipe z uses roughly the same scale as x
        return FaceLandmarks(points=points, image_shape=(height, width))

    # -------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if self._impl is not None:
            try:
                self._impl.close()
            except Exception:  # pragma: no cover - backend dependent
                pass
            self._impl = None

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
