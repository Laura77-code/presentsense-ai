"""Face Mesh landmark utilities for PresentSense Phase 3.5.

The values produced here are lightweight geometric cues for presentation
practice. They are not gaze tracking, psychological inference, or full-body
posture estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import mediapipe as mp
import numpy as np


@dataclass
class FaceLandmarkMetrics:
    """Per-frame facial landmark geometry cues.

    All coordinates are normalized to the image size where possible.
    yaw_proxy is not a true angle; it is a normalized left/right face-orientation
    proxy based on the nose position relative to the eye midpoint.
    """

    detected: bool
    nose_x_norm: float = 0.0
    nose_y_norm: float = 0.0
    face_center_x_norm: float = 0.0
    face_center_y_norm: float = 0.0
    yaw_proxy: float = 0.0
    roll_deg: float = 0.0
    mouth_open_ratio: float = 0.0
    eye_open_ratio: float = 0.0
    face_height_norm: float = 0.0


class MediaPipeFaceMeshAnalyzer:
    """Extract simple geometric cues from MediaPipe Face Mesh landmarks."""

    # Landmark indices from MediaPipe Face Mesh topology.
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    NOSE_TIP = 1
    MOUTH_UPPER = 13
    MOUTH_LOWER = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    FOREHEAD = 10
    CHIN = 152

    def __init__(self, max_num_faces: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def analyze(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarkMetrics]:
        """Return Face Mesh-derived cues for one frame, or None if no face exists."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        height, width = frame_bgr.shape[:2]
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        results = self._mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        def pt(idx: int) -> np.ndarray:
            lm = landmarks[idx]
            return np.array([lm.x, lm.y], dtype=np.float32)

        left_outer = pt(self.LEFT_EYE_OUTER)
        left_inner = pt(self.LEFT_EYE_INNER)
        right_inner = pt(self.RIGHT_EYE_INNER)
        right_outer = pt(self.RIGHT_EYE_OUTER)
        left_eye_top = pt(self.LEFT_EYE_TOP)
        left_eye_bottom = pt(self.LEFT_EYE_BOTTOM)
        right_eye_top = pt(self.RIGHT_EYE_TOP)
        right_eye_bottom = pt(self.RIGHT_EYE_BOTTOM)
        nose = pt(self.NOSE_TIP)
        mouth_upper = pt(self.MOUTH_UPPER)
        mouth_lower = pt(self.MOUTH_LOWER)
        mouth_left = pt(self.MOUTH_LEFT)
        mouth_right = pt(self.MOUTH_RIGHT)
        forehead = pt(self.FOREHEAD)
        chin = pt(self.CHIN)

        eye_mid = (left_outer + right_outer) / 2.0
        eye_width = float(np.linalg.norm(right_outer - left_outer))
        eye_width = max(eye_width, 1e-6)

        # Nose horizontal displacement relative to eye midpoint. This is a proxy
        # for left/right head orientation, not a true yaw angle.
        yaw_proxy = float((nose[0] - eye_mid[0]) / eye_width)

        dy, dx = right_outer[1] - left_outer[1], right_outer[0] - left_outer[0]
        roll_deg = float(np.degrees(np.arctan2(dy, dx)))

        face_height = float(np.linalg.norm(chin - forehead))
        face_height = max(face_height, 1e-6)
        mouth_open_ratio = float(np.linalg.norm(mouth_lower - mouth_upper) / face_height)

        left_eye_height = float(np.linalg.norm(left_eye_top - left_eye_bottom))
        right_eye_height = float(np.linalg.norm(right_eye_top - right_eye_bottom))
        left_eye_width = float(np.linalg.norm(left_inner - left_outer))
        right_eye_width = float(np.linalg.norm(right_outer - right_inner))
        eye_open_ratio = float(((left_eye_height / max(left_eye_width, 1e-6)) + (right_eye_height / max(right_eye_width, 1e-6))) / 2.0)

        # Robust-ish center from all landmarks.
        xs = np.array([lm.x for lm in landmarks], dtype=np.float32)
        ys = np.array([lm.y for lm in landmarks], dtype=np.float32)
        center_x = float(np.mean(xs))
        center_y = float(np.mean(ys))

        return FaceLandmarkMetrics(
            detected=True,
            nose_x_norm=float(nose[0]),
            nose_y_norm=float(nose[1]),
            face_center_x_norm=center_x,
            face_center_y_norm=center_y,
            yaw_proxy=yaw_proxy,
            roll_deg=roll_deg,
            mouth_open_ratio=mouth_open_ratio,
            eye_open_ratio=eye_open_ratio,
            face_height_norm=face_height,
        )

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> "MediaPipeFaceMeshAnalyzer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()
