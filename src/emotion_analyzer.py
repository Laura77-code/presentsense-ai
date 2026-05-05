"""Emotion inference utilities for cropped face images."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.dataset import FER2013_CLASSES
from src.model import load_model_checkpoint


@dataclass
class EmotionPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


def crop_face_with_margin(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], margin_ratio: float = 0.20) -> np.ndarray:
    """Crop a face from a frame with a small margin around the detected box."""
    x, y, w, h = bbox
    height, width = frame_bgr.shape[:2]
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(width, x + w + margin_x)
    y2 = min(height, y + h + margin_y)
    return frame_bgr[y1:y2, x1:x2]


class EmotionAnalyzer:
    """Load a trained emotion model and predict smoothed emotions over time."""

    def __init__(self, checkpoint_path: str | Path, device: Optional[str] = None, smoothing_window: int = 5) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.checkpoint = load_model_checkpoint(checkpoint_path, device=self.device)
        self.class_names: list[str] = self.checkpoint.get("class_names", FER2013_CLASSES)
        self.history: deque[np.ndarray] = deque(maxlen=max(1, smoothing_window))
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def predict(self, face_bgr: np.ndarray) -> Optional[EmotionPrediction]:
        if face_bgr is None or face_bgr.size == 0:
            return None

        face_rgb = cv.cvtColor(face_bgr, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        self.history.append(probs)
        smoothed = np.mean(np.stack(list(self.history), axis=0), axis=0)
        idx = int(np.argmax(smoothed))
        label = self.class_names[idx]
        confidence = float(smoothed[idx])
        probabilities = {name: float(smoothed[i]) for i, name in enumerate(self.class_names)}
        return EmotionPrediction(label=label, confidence=confidence, probabilities=probabilities)
