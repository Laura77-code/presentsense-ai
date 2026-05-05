"""OpenCV visualization helpers for PresentSense."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2 as cv
import numpy as np

from src.face_detector import FaceDetectionResult


@dataclass
class FPSCounter:
    """Simple FPS counter with exponential smoothing."""

    smoothing: float = 0.9
    _last_time: float = field(default_factory=time.perf_counter)
    _fps: float = 0.0

    def update(self) -> float:
        current_time = time.perf_counter()
        elapsed = current_time - self._last_time
        self._last_time = current_time

        if elapsed <= 0:
            return self._fps

        current_fps = 1.0 / elapsed
        if self._fps == 0.0:
            self._fps = current_fps
        else:
            self._fps = self.smoothing * self._fps + (1.0 - self.smoothing) * current_fps
        return self._fps


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    """Draw readable text with a dark shadow background effect."""
    x, y = origin
    cv.putText(frame, text, (x + 1, y + 1), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv.LINE_AA)
    cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv.LINE_AA)


def draw_face_overlay(frame: np.ndarray, detection: Optional[FaceDetectionResult]) -> None:
    """Draw a face bounding box and detection status."""
    if detection is None:
        draw_text(frame, "No face detected", (20, 40), color=(0, 0, 255), scale=0.8)
        return

    x, y, w, h = detection.bbox
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"Face detected: {detection.confidence:.2f}"
    label_y = max(25, y - 10)
    draw_text(frame, label, (x, label_y), color=(0, 255, 0), scale=0.6)


def draw_emotion_overlay(frame: np.ndarray, label: str, confidence: float, origin: tuple[int, int] = (20, 75)) -> None:
    """Draw current emotion prediction on the frame."""
    draw_text(frame, f"Emotion: {label} ({confidence:.2f})", origin, color=(0, 255, 255), scale=0.7)


def draw_base_overlay(
    frame: np.ndarray,
    frame_number: int,
    fps: float,
    source_label: str,
    phase_label: str = "Phase 2: Face + Emotion Pipeline",
) -> None:
    """Draw video metadata overlays."""
    height, _ = frame.shape[:2]
    draw_text(frame, f"PresentSense | {phase_label}", (20, height - 70), color=(255, 255, 255), scale=0.65)
    draw_text(frame, f"Source: {source_label}", (20, height - 45), color=(255, 255, 255), scale=0.6)
    draw_text(frame, f"Frame: {frame_number} | FPS: {fps:.1f}", (20, height - 20), color=(255, 255, 255), scale=0.6)
