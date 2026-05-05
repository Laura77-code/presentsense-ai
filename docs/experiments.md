# Phase 2 Experiments

PresentSense Phase 2 trains facial emotion recognition models on FER2013.

## Dataset

FER2013 contains seven expression labels:

```text
angry, disgust, fear, happy, sad, surprise, neutral
```

Supported layouts:

```text
data/fer2013/fer2013.csv
```

or:

```text
data/fer2013/train/<class_name>/
data/fer2013/test/<class_name>/
```

## Experiment 1: Custom CNN from Scratch

Command:

```powershell
py train.py --experiment exp01_custom_cnn --epochs 10 --batch-size 64
```

Purpose:

- Provide a simple CNN baseline.
- Show understanding of convolutional layers, pooling, and classification heads.
- Compare a model trained from scratch with transfer learning models.

## Experiment 2: ResNet18 Frozen Backbone

Command:

```powershell
py train.py --experiment exp02_resnet18_frozen --epochs 10 --batch-size 64
```

Purpose:

- Use ImageNet pretrained features.
- Freeze the backbone.
- Train only the final fully connected layer.
- Fast and stable baseline for a student project.

## Experiment 3: ResNet18 Fine-Tune Last Layers

Command:

```powershell
py train.py --experiment exp03_resnet18_finetune --epochs 10 --batch-size 64
```

Purpose:

- Fine-tune the deeper ResNet layers and the classification head.
- Adapt pretrained visual features to facial expression recognition.

## Experiment 4: MobileNetV3 Fine-Tune Classifier

Command:

```powershell
py train.py --experiment exp04_mobilenet_finetune --epochs 10 --batch-size 64
```

Purpose:

- Test a lightweight model suitable for real-time webcam inference.
- Compare speed-oriented architecture with ResNet18.

## Metrics

Each experiment reports:

- Train loss.
- Validation loss.
- Train accuracy.
- Validation accuracy.
- Test accuracy.
- Macro precision.
- Macro recall.
- Macro F1-score.
- Confusion matrix.

## Outputs

```text
models/best_<experiment>.pth
outputs/charts/<experiment>_training_curves.png
outputs/charts/<experiment>_confusion_matrix.png
outputs/reports/experiments.csv
```

## Quick Test

Before running full training, verify the pipeline with:

```powershell
py train.py --experiment exp02_resnet18_frozen --epochs 1 --batch-size 16 --quick-test
```
