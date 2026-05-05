"""FER2013 dataset loading utilities for PresentSense Phase 2.

Supported layouts:
1) CSV format:
   data/fer2013/fer2013.csv
   Expected columns: emotion, pixels, Usage

2) Folder format:
   data/fer2013/train/<class_name>/*.png
   data/fer2013/test/<class_name>/*.png

The code auto-detects the available format and returns PyTorch DataLoaders.
"""

from __future__ import annotations

# pandas is imported lazily only when CSV loading is needed.
pd = None

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

FER2013_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FER2013_INDEX_TO_CLASS = {idx: name for idx, name in enumerate(FER2013_CLASSES)}
FER2013_CLASS_TO_INDEX = {name: idx for idx, name in enumerate(FER2013_CLASSES)}


@dataclass(frozen=True)
class DatasetInfo:
    """Small metadata object returned with the data loaders."""

    format_name: str
    class_to_idx: Dict[str, int]
    train_size: int
    val_size: int
    test_size: int


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train/eval preprocessing pipelines.

    FER2013 images are grayscale 48x48. For pretrained ImageNet models, images are
    converted to RGB, resized to 224x224, and normalized using ImageNet stats.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_tf, eval_tf


class FER2013CSVDataset(Dataset):
    """PyTorch Dataset for the classic FER2013 CSV format."""

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[transforms.Compose] = None) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[index]
        label = int(row["emotion"])
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        image = pixels.reshape(48, 48)
        pil_image = Image.fromarray(image, mode="L")

        if self.transform is not None:
            pil_image = self.transform(pil_image)

        return pil_image, label


def _quick_subset(dataset: Dataset, max_samples: Optional[int], seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def _split_csv_by_usage(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    required = {"emotion", "pixels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FER2013 CSV is missing columns: {sorted(missing)}")

    if "Usage" in df.columns:
        train_df = df[df["Usage"].str.lower() == "training"]
        val_df = df[df["Usage"].str.lower() == "publictest"]
        test_df = df[df["Usage"].str.lower() == "privatetest"]
        if len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0:
            return train_df, val_df, test_df

    # Fallback for CSVs without Usage: deterministic 80/10/10 split.
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def _build_csv_loaders(
    csv_path: Path,
    batch_size: int,
    image_size: int,
    num_workers: int,
    quick_test: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    train_tf, eval_tf = build_transforms(image_size=image_size)
    train_df, val_df, test_df = _split_csv_by_usage(csv_path)

    if quick_test:
        train_df = train_df.head(512)
        val_df = val_df.head(128)
        test_df = test_df.head(128)

    train_ds = FER2013CSVDataset(train_df, transform=train_tf)
    val_ds = FER2013CSVDataset(val_df, transform=eval_tf)
    test_ds = FER2013CSVDataset(test_df, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    info = DatasetInfo(
        format_name="csv",
        class_to_idx=FER2013_CLASS_TO_INDEX,
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
    )
    return train_loader, val_loader, test_loader, info


def _build_folder_loaders(
    data_dir: Path,
    batch_size: int,
    image_size: int,
    num_workers: int,
    quick_test: bool,
    seed: int,
    val_ratio: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    train_tf, eval_tf = build_transforms(image_size=image_size)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Folder format requires data/fer2013/train and data/fer2013/test")

    full_train = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    val_size = max(1, int(len(full_train) * val_ratio))
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_indices = random_split(full_train, [train_size, val_size], generator=generator)

    # Use eval transforms for validation by creating a separate dataset and reusing indices.
    full_train_eval = datasets.ImageFolder(train_dir, transform=eval_tf)
    val_ds = Subset(full_train_eval, val_indices.indices)

    if quick_test:
        train_ds = _quick_subset(train_ds, 512, seed)
        val_ds = _quick_subset(val_ds, 128, seed)
        test_ds = _quick_subset(test_ds, 128, seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    info = DatasetInfo(
        format_name="folder",
        class_to_idx=full_train.class_to_idx,
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
    )
    return train_loader, val_loader, test_loader, info


def get_fer2013_loaders(
    data_dir: str | Path = "data/fer2013",
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 0,
    quick_test: bool = False,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    """Auto-detect FER2013 format and return train/val/test loaders."""
    data_path = Path(data_dir)
    csv_path = data_path / "fer2013.csv"
    train_dir = data_path / "train"
    test_dir = data_path / "test"

    if csv_path.exists():
        return _build_csv_loaders(csv_path, batch_size, image_size, num_workers, quick_test, seed)

    if train_dir.exists() and test_dir.exists():
        return _build_folder_loaders(data_path, batch_size, image_size, num_workers, quick_test, seed)

    raise FileNotFoundError(
        "FER2013 dataset was not found. Expected one of these formats:\n"
        "1) data/fer2013/fer2013.csv\n"
        "2) data/fer2013/train/<class_name>/ and data/fer2013/test/<class_name>/\n"
        "Download FER2013 manually from Kaggle and place it in data/fer2013/."
    )
