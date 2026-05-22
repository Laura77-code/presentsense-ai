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
    """Draw readable text with a dark shadow effect."""
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


def draw_expression_overlay(
    frame: np.ndarray,
    display_label: str,
    confidence: float,
    top_k: Optional[list[tuple[str, float]]] = None,
    origin: tuple[int, int] = (20, 75),
) -> None:
    """Draw current facial-expression cue and optional top-3 probabilities."""
    color = (0, 255, 255) if display_label != "uncertain" else (0, 165, 255)
    draw_text(frame, f"Expression: {display_label} ({confidence:.2f})", origin, color=color, scale=0.65)
    if top_k:
        top_text = "Top-3: " + " | ".join([f"{name} {prob:.2f}" for name, prob in top_k])
        draw_text(frame, top_text, (origin[0], origin[1] + 26), color=(255, 255, 0), scale=0.45, thickness=1)


def draw_emotion_overlay(frame: np.ndarray, label: str, confidence: float, origin: tuple[int, int] = (20, 75)) -> None:
    """Backward-compatible wrapper for older Phase 2 code."""
    draw_expression_overlay(frame, label, confidence, top_k=None, origin=origin)


def draw_metric_overlay(
    frame: np.ndarray,
    looking_forward: Optional[bool],
    movement_level: str,
    elapsed_time: float,
    yaw_proxy: float = 0.0,
    mouth_open_ratio: float = 0.0,
    face_mesh_detected: bool = False,
) -> None:
    """Draw Phase 3.5 live metric hints."""
    if looking_forward is None:
        gaze_text = "Looking-forward approx: no face"
        gaze_color = (0, 0, 255)
    else:
        gaze_text = "Looking-forward approx: yes" if looking_forward else "Looking-forward approx: no"
        gaze_color = (0, 255, 0) if looking_forward else (0, 165, 255)

    mesh_text = "Face Mesh: yes" if face_mesh_detected else "Face Mesh: fallback"
    draw_text(frame, gaze_text, (20, 40), color=gaze_color, scale=0.50)
    draw_text(frame, f"Head/face movement: {movement_level}", (20, 63), color=(255, 255, 255), scale=0.50)
    draw_text(frame, mesh_text, (20, 86), color=(255, 255, 255), scale=0.50)
    if face_mesh_detected:
        draw_text(frame, f"Yaw proxy: {yaw_proxy:+.2f} | Mouth: {mouth_open_ratio:.3f}", (20, 109), color=(255, 255, 255), scale=0.45, thickness=1)
        draw_text(frame, f"Elapsed: {elapsed_time:.1f}s", (20, 132), color=(255, 255, 255), scale=0.50)
    else:
        draw_text(frame, f"Elapsed: {elapsed_time:.1f}s", (20, 109), color=(255, 255, 255), scale=0.50)


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
