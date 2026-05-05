"""Model definitions and checkpoint loading for PresentSense Phase 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torchvision import models

from src.dataset import FER2013_CLASSES


class CustomEmotionCNN(nn.Module):
    """A compact CNN baseline for FER2013 emotion classification."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def create_model(experiment: str, num_classes: int = 7) -> nn.Module:
    """Create one of the Phase 2 experiment models."""
    experiment = experiment.lower()

    if experiment == "exp01_custom_cnn":
        return CustomEmotionCNN(num_classes=num_classes)

    if experiment == "exp02_resnet18_frozen":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        _freeze(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if experiment == "exp03_resnet18_finetune":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        # Freeze early layers, fine-tune layer4 and classification head.
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("layer4") or name.startswith("fc")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if experiment == "exp04_mobilenet_finetune":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        # Freeze feature extractor, train only the classifier for a fast student MVP.
        _freeze(model.features)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unknown experiment '{experiment}'. Choose one of: "
        "exp01_custom_cnn, exp02_resnet18_frozen, exp03_resnet18_finetune, exp04_mobilenet_finetune"
    )


def save_model_checkpoint(
    path: str | Path,
    model: nn.Module,
    experiment: str,
    class_names: list[str] | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint with enough metadata for inference."""
    checkpoint = {
        "experiment": experiment,
        "class_names": class_names or FER2013_CLASSES,
        "state_dict": model.state_dict(),
    }
    if extra:
        checkpoint.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_model_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved emotion model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    experiment = checkpoint.get("experiment", "exp02_resnet18_frozen")
    class_names = checkpoint.get("class_names", FER2013_CLASSES)
    model = create_model(experiment=experiment, num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
