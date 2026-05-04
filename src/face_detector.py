"""Face detection utilities for PresentSense Phase 1.

This module wraps MediaPipe Face Detection and exposes a small, clean API
that can be used by webcam and video-processing scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2 as cv
import mediapipe as mp
import numpy as np


@dataclass
class FaceDetectionResult:
    """Single-face detection result in pixel coordinates."""

    bbox: Tuple[int, int, int, int]
    confidence: float

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def w(self) -> int:
        return self.bbox[2]

    @property
    def h(self) -> int:
        return self.bbox[3]


class MediaPipeFaceDetector:
    """Detects faces using MediaPipe Face Detection.

    For Phase 1, we keep only the highest-confidence face because the project
    is focused on a single presenter in a classroom presentation video.
    """

    def __init__(self, model_selection: int = 0, min_detection_confidence: float = 0.5) -> None:
        self._mp_face_detection = mp.solutions.face_detection
        self._detector = self._mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceDetectionResult]:
        """Return the best detected face in BGR frame, or None if no face exists."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        results = self._detector.process(frame_rgb)

        if not results.detections:
            return None

        height, width = frame_bgr.shape[:2]
        best_detection = max(results.detections, key=lambda detection: detection.score[0])
        relative_box = best_detection.location_data.relative_bounding_box

        x = int(relative_box.xmin * width)
        y = int(relative_box.ymin * height)
        w = int(relative_box.width * width)
        h = int(relative_box.height * height)

        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        return FaceDetectionResult(bbox=(x, y, w, h), confidence=float(best_detection.score[0]))

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()

    def __enter__(self) -> "MediaPipeFaceDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()
