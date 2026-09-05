"""Stress detection from facial micro-expressions.

Modules
-------
camera         camera-agnostic capture (USB index or RTSP URL)
face_mesh      MediaPipe FaceMesh wrapper, canonical alignment, FACS geometry
optical_flow   Farneback micro-motion extraction and burst detection
au_estimator   Action Unit intensity estimation (rule-based, FACS-inspired)
stress_scorer  Action Units -> stress index (0-10)
visualizer     real-time overlay
pipeline       wiring of all of the above into one per-frame call
"""

__version__ = "0.1.0"
__all__ = [
    "camera",
    "face_mesh",
    "optical_flow",
    "au_estimator",
    "stress_scorer",
    "visualizer",
    "pipeline",
]
