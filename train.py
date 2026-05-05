"""Train FER2013 emotion recognition models for PresentSense Phase 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.dataset import FER2013_CLASSES, get_fer2013_loaders
from src.model import create_model, save_model_checkpoint
from src.train_utils import (
    append_experiment_result,
    evaluate,
    get_device,
    plot_confusion_matrix,
    plot_training_curves,
    set_seed,
    train_one_epoch,
)

EXPERIMENTS = [
    "exp01_custom_cnn",
    "exp02_resnet18_frozen",
    "exp03_resnet18_finetune",
    "exp04_mobilenet_finetune",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PresentSense emotion recognition models on FER2013")
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--data-dir", type=Path, default=Path("data/fer2013"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows for fewer multiprocessing issues")
    parser.add_argument("--quick-test", action="store_true", help="Use a small subset to verify that training works")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, data_info = get_fer2013_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=224,
        num_workers=args.num_workers,
        quick_test=args.quick_test,
        seed=args.seed,
    )
    print(f"Dataset format: {data_info.format_name}")
    print(f"Train/Val/Test: {data_info.train_size}/{data_info.val_size}/{data_info.test_size}")

    model = create_model(args.experiment, num_classes=len(FER2013_CLASSES)).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    checkpoint_path = Path("models") / f"best_{args.experiment}.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        print(f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} macro_f1={val_metrics['macro_f1']:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model_checkpoint(
                checkpoint_path,
                model,
                experiment=args.experiment,
                class_names=FER2013_CLASSES,
                extra={"best_val_accuracy": best_val_acc, "epoch": epoch},
            )
            print(f"Saved best checkpoint: {checkpoint_path}")

    curves_path = Path("outputs/charts") / f"{args.experiment}_training_curves.png"
    plot_training_curves(history, curves_path)
    print(f"Training curves saved to: {curves_path}")

    # Reload best checkpoint for final test evaluation.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_acc, test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    cm_path = Path("outputs/charts") / f"{args.experiment}_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, FER2013_CLASSES, cm_path)
    print(f"Confusion matrix saved to: {cm_path}")

    result_row = {
        "experiment": args.experiment,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dataset_format": data_info.format_name,
        "train_size": data_info.train_size,
        "val_size": data_info.val_size,
        "test_size": data_info.test_size,
        "best_val_accuracy": f"{best_val_acc:.4f}",
        "test_accuracy": f"{test_acc:.4f}",
        "macro_precision": f"{test_metrics['macro_precision']:.4f}",
        "macro_recall": f"{test_metrics['macro_recall']:.4f}",
        "macro_f1": f"{test_metrics['macro_f1']:.4f}",
        "checkpoint": str(checkpoint_path),
    }
    experiments_csv = Path("outputs/reports/experiments.csv")
    append_experiment_result(experiments_csv, result_row)

    print("\nFinal test metrics")
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    print(f"macro_precision={test_metrics['macro_precision']:.4f}")
    print(f"macro_recall={test_metrics['macro_recall']:.4f}")
    print(f"macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"Experiment row appended to: {experiments_csv}")


if __name__ == "__main__":
    main()
