# PresentSense: A Computer Vision-Based Presentation Coach

## Overview

**PresentSense** is a Computer Vision final project that analyzes a presentation video or webcam stream and provides visual feedback for presentation practice.

The system detects the presenter's face, estimates model-predicted visible facial expression cues, analyzes Face Mesh geometry, approximates looking-forward behavior, measures head/face stability, and generates a presentation feedback report with charts and recommendations.

> **Important:** PresentSense is **not** a medical, psychological, or clinical diagnosis tool. It does not measure true emotions, confidence, personality, mental health, or presentation quality absolutely. It is a visual communication feedback prototype for student presentation practice.

---

## Current Status

| Phase | Status | Description |
|---|---|---|
| Phase 1 | Completed | OpenCV + MediaPipe webcam/video face detection pipeline |
| Phase 2 | Completed | FER2013 expression recognition training and webcam/video inference |
| Phase 3 | Completed / polishing | Face Mesh visual metrics, looking-forward approximation, head/face stability, expression uncertainty handling, charts, and report generation |
| Phase 4 | Planned | Streamlit app, dashboard, downloadable reports, and final GitHub polish |

---

## Motivation

Students often practice presentations alone and receive feedback only after presenting in class. PresentSense helps students review visual communication cues from a practice recording, such as face visibility, looking-forward behavior, head/face stability, visible facial cue variation, and camera presence.

The goal is to provide supportive feedback for improving presentation delivery, not to judge personality, confidence, or real emotion.

---

## Computer Vision Concepts Used

This project connects directly with core Computer Vision topics:

- OpenCV video input and output.
- Frame-by-frame video processing.
- OpenCV drawing functions and overlays.
- MediaPipe face detection.
- MediaPipe Face Mesh landmark detection.
- Face cropping and preprocessing.
- Image resizing and normalization.
- Transfer learning for image classification.
- CNN-based facial expression recognition.
- PyTorch model training and inference.
- Softmax confidence and uncertainty thresholding.
- Temporal smoothing of frame-level predictions.
- Geometric landmark analysis.
- Time-series aggregation of visual cues.
- Confusion matrix and classification metrics.
- Chart generation with Matplotlib.
- Markdown/JSON/CSV report generation.

---

## Features

### Phase 1: Face Detection Pipeline

- Webcam input.
- Local video file input.
- MediaPipe face detection.
- Face bounding box overlay.
- Face detection confidence overlay.
- FPS and frame number overlay.
- Status text when no face is detected.
- Annotated video export.

### Phase 2: Expression Recognition

- FER2013 dataset loader.
- Support for folder and CSV dataset formats.
- Transfer learning experiments:
  - Custom CNN baseline.
  - ResNet18 frozen backbone.
  - ResNet18 fine-tuning.
  - MobileNetV3 fine-tuning.
- Training and validation curves.
- Confusion matrix generation.
- Experiment metrics exported to CSV.
- Best model checkpoint saving.
- Face crop preprocessing for model inference.
- Webcam/video expression inference.
- Rolling probability smoothing.
- Uncertainty handling for low-confidence predictions.

### Phase 3: Presentation Visual Metrics

- Face Mesh landmark extraction.
- Model-predicted visible expression cue timeline.
- Expression distribution with `uncertain` class.
- Looking-forward approximation.
- Head/face stability analysis.
- Mouth openness over time.
- Visible expression variation score.
- Expression variety score.
- Per-frame metric export.
- Summary JSON report.
- Markdown presentation feedback report.
- Recommendation generation.
- Charts exported to `outputs/charts/`.

---

## Demo

### Phase 1: Face Detection

Phase 1 validates the real-time OpenCV/MediaPipe video pipeline.

![Phase 1 Webcam Face Detection](outputs/screenshots/phase1/phase1_webcam_face_detection.png)

Optional demo video:

[Watch Phase 1 Webcam Demo](outputs/videos/phase1_webcam_demo.mp4)

---

### Phase 2: Face + Expression Recognition

Phase 2 adds a FER2013-trained expression recognition model. The model predicts visible facial expression cues from the detected face crop.

#### Happy Example

![Phase 2 Happy Expression](outputs/screenshots/phase2/happy.png)

#### Sad Example

![Phase 2 Sad Expression](outputs/screenshots/phase2/sad.png)

#### Known Limitation Example

The classifier can confuse expressions in real webcam conditions because FER2013 differs from live camera footage in lighting, resolution, pose, and expression intensity.

![Phase 2 Known Error](outputs/screenshots/phase2/neutral_error.png)

Optional demo video:

[Watch Phase 2 ResNet18 Expression Demo](outputs/videos/phase2_resnet18_emotion_demo.mp4)

---

### Phase 3: Visual Metrics and Report Generation

Phase 3 combines expression inference with Face Mesh geometry and temporal aggregation.

Example Phase 3.5 demo results:

| Metric | Value |
|---|---:|
| Duration | 19.2 seconds |
| Analyzed frames | 576 |
| Face detection rate | 100.0% |
| Face Mesh detection rate | 100.0% |
| Average model confidence | 0.6727 |
| Looking-forward approximation | 84.9 / 100 |
| Visible expression variation | 55.8 / 100 |
| Head/face stability | 83.76 / 100 |
| Expression variety | 39.45 / 100 |
| Overall visual score | 71.34 / 100 |

The dominant model-predicted visible cue was `uncertain`, which means many frames were below the confidence threshold. This is intentional: the system avoids forcing expression labels when the classifier is not confident enough.

#### Expression Distribution

![Phase 3 Expression Distribution](outputs/charts/phase3/phase3_expression_distribution.png)

#### Expression Timeline

![Phase 3 Expression Timeline](outputs/charts/phase3/phase3_expression_timeline.png)

#### Looking-Forward Approximation

![Phase 3 Looking Forward](outputs/charts/phase3/phase3_looking_forward_over_time.png)

#### Head/Face Movement

![Phase 3 Head Face Movement](outputs/charts/phase3/phase3_head_face_movement_over_time.png)

#### Mouth Openness

![Phase 3 Mouth Openness](outputs/charts/phase3/phase3_mouth_openness_over_time.png)

Example generated reports:

```text
outputs/reports/phase3/presentation_report_example.md
outputs/reports/phase3/summary_report_example.json
```

Optional demo video:

[Watch Phase 3 Visual Metrics Demo](outputs/videos/phase3_visual_metrics_demo.mp4)

> If videos are not committed because of file size, run the demo locally using the commands below.

---

## Dataset

The expression recognition model uses **FER2013** from Kaggle: *Facial Expression Recognition Challenge*.

Expected classes:

```text
angry
disgust
fear
happy
neutral
sad
surprise
```

The dataset is not included in this repository. Do not commit the dataset to GitHub.

### Option A: Folder Format

```text
data/fer2013/train/angry/
data/fer2013/train/disgust/
data/fer2013/train/fear/
data/fer2013/train/happy/
data/fer2013/train/neutral/
data/fer2013/train/sad/
data/fer2013/train/surprise/

data/fer2013/test/angry/
data/fer2013/test/disgust/
data/fer2013/test/fear/
data/fer2013/test/happy/
data/fer2013/test/neutral/
data/fer2013/test/sad/
data/fer2013/test/surprise/
```

### Option B: CSV Format

```text
data/fer2013/fer2013.csv
```

Expected CSV columns:

```text
emotion,pixels,Usage
```

---

## Installation

### Windows PowerShell

Recommended location:

```powershell
C:\cv\presentsense
```

Create and activate the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows Compatibility Notes

The project pins versions to avoid common MediaPipe/OpenCV/Numpy issues on Windows:

```text
numpy==1.26.4
opencv-contrib-python==4.11.0.86
mediapipe==0.10.21
protobuf==4.25.9
```

---

## Optional GPU Setup

If you have an NVIDIA GPU, install PyTorch with CUDA support.

Example for CUDA 12.6 wheels:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

---

## Running Phase 1

### Webcam

```powershell
python analyze_video.py --source webcam
```

### Webcam with saved video

```powershell
python analyze_video.py --source webcam --output outputs/videos/phase1_webcam_demo.mp4
```

### Local video

```powershell
python analyze_video.py --source data/samples/presentation.mp4 --output outputs/videos/phase1_video_demo.mp4
```

---

## Training Phase 2 Models

First, place FER2013 in `data/fer2013/`.

### Quick Test

```powershell
python train.py --experiment exp02_resnet18_frozen --epochs 1 --batch-size 16 --quick-test
```

### Full Experiments

```powershell
python train.py --experiment exp01_custom_cnn --epochs 10 --batch-size 32
python train.py --experiment exp02_resnet18_frozen --epochs 10 --batch-size 32
python train.py --experiment exp03_resnet18_finetune --epochs 10 --batch-size 16
python train.py --experiment exp04_mobilenet_finetune --epochs 20 --batch-size 32
```

The best model from the completed experiments was:

```text
models/best_exp03_resnet18_finetune.pth
```

Model checkpoints are ignored by Git by default because they can be large.

---

## Running Phase 2 Expression Demo

```powershell
python analyze_video.py --source webcam --model models/best_exp03_resnet18_finetune.pth
```

Save an annotated video:

```powershell
python analyze_video.py --source webcam --model models/best_exp03_resnet18_finetune.pth --output outputs/videos/phase2_resnet18_emotion_demo.mp4
```

---

## Running Phase 3 Visual Metrics

Run full visual analysis with expression inference, Face Mesh metrics, charts, and reports:

```powershell
python analyze_video.py --source webcam --model models/best_exp03_resnet18_finetune.pth
```

Save a Phase 3 demo video:

```powershell
python analyze_video.py --source webcam --model models/best_exp03_resnet18_finetune.pth --output outputs/videos/phase3_visual_metrics_demo.mp4
```

Analyze a local video:

```powershell
python analyze_video.py --source data/samples/presentation.mp4 --model models/best_exp03_resnet18_finetune.pth --output outputs/videos/phase3_video_demo.mp4
```

If Face Mesh is slow, disable it:

```powershell
python analyze_video.py --source webcam --model models/best_exp03_resnet18_finetune.pth --no-face-mesh
```

Adjust uncertainty threshold:

```powershell
python analyze_video.py --source webcam --model models/best_exp03_resnet18_finetune.pth --uncertainty-threshold 0.55
python analyze_video.py --source webcam --model models/best_exp03_resnet18_finetune.pth --uncertainty-threshold 0.65
```

Generated files:

```text
outputs/reports/frame_metrics.csv
outputs/reports/summary_report.json
outputs/reports/presentation_report.md
outputs/charts/phase3_expression_distribution.png
outputs/charts/phase3_expression_timeline.png
outputs/charts/phase3_looking_forward_over_time.png
outputs/charts/phase3_head_face_movement_over_time.png
outputs/charts/phase3_mouth_openness_over_time.png
```

Curated example assets are stored in:

```text
outputs/charts/phase3/
outputs/reports/phase3/
outputs/screenshots/phase3/
```

---

## Methodology

### Face Detection

MediaPipe detects the presenter's face in each frame. OpenCV draws the bounding box, confidence, FPS, and status overlays.

### Expression Recognition

1. The detected face is cropped with a margin.
2. The crop is resized to `224 x 224`.
3. The crop is normalized using ImageNet statistics.
4. A PyTorch model predicts FER2013 expression classes.
5. Softmax probabilities are smoothed over time.
6. If confidence is below the threshold, the frame is labeled as `uncertain`.

### Face Mesh Geometry

Face Mesh landmarks are used to estimate geometric presentation cues:

- Nose and eye symmetry for looking-forward approximation.
- Face roll angle proxy.
- Nose/landmark movement for head/face stability.
- Mouth openness ratio over time.
- Eye openness variation as part of expression variation.

### Looking-Forward Approximation

This is a heuristic based on face position, Face Mesh nose/eye symmetry, and face roll. It is **not** real gaze tracking and does not guarantee actual eye contact.

### Head/Face Stability

This score uses bounding-box movement and Face Mesh landmark movement. It estimates whether the face remains stable in the camera frame. It is **not** full-body posture analysis.

### Visible Expression Variation

This score combines model-predicted expression probability variation with Face Mesh mouth/eye movement. It estimates variation in visible facial cues, not real emotional state.

### Overall Score

The final score is a heuristic weighted score:

```text
overall_score =
  0.35 * looking_forward_score
+ 0.30 * visible_expression_variation_score
+ 0.25 * head_face_stability_score
+ 0.10 * expression_variety_score
```

---

## Experiments and Results

The experiment results are saved in:

```text
outputs/reports/experiments.csv
```

| Experiment | Model | Epochs | Batch Size | Test Accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|
| exp04 | MobileNetV3 fine-tune | 10 | 32 | 49.53% | 42.74% |
| exp04 | MobileNetV3 fine-tune | 20 | 32 | 51.49% | 45.62% |
| exp03 | ResNet18 fine-tune | 10 | 16 | **64.11%** | **60.87%** |

The best Phase 2 model was **ResNet18 fine-tuned on FER2013**, reaching **64.11% test accuracy** and **60.87% macro F1**.

### ResNet18 Fine-Tune Training Curves

![ResNet18 Fine-Tune Training Curves](outputs/charts/exp03_resnet18_finetune_training_curves.png)

### ResNet18 Fine-Tune Confusion Matrix

![ResNet18 Fine-Tune Confusion Matrix](outputs/charts/exp03_resnet18_finetune_confusion_matrix.png)

### MobileNetV3 Fine-Tune Training Curves

![MobileNetV3 Fine-Tune Training Curves](outputs/charts/exp04_mobilenet_finetune_training_curves.png)

### MobileNetV3 Fine-Tune Confusion Matrix

![MobileNetV3 Fine-Tune Confusion Matrix](outputs/charts/exp04_mobilenet_finetune_confusion_matrix.png)

---

## Phase 3 Report Example

Latest Phase 3.5 example:

| Metric | Value |
|---|---:|
| Duration | 19.2 seconds |
| Face detection rate | 100.0% |
| Face Mesh detection rate | 100.0% |
| Dominant model-predicted cue | uncertain |
| Average model confidence | 0.6727 |
| Happy cue percentage | 47.05% |
| Uncertain percentage | 48.44% |
| Looking-forward approximation | 84.9 / 100 |
| Visible expression variation | 55.8 / 100 |
| Head/face stability | 83.76 / 100 |
| Expression variety | 39.45 / 100 |
| Final overall score | 71.34 / 100 |

Interpretation: the system detected the face and Face Mesh reliably. Expression recognition remained uncertain for many frames, so the report treats expression labels as model-predicted visual cues rather than real emotions.

---

## Repository Structure

```text
presentsense/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── app.py
├── train.py
├── analyze_video.py
├── config.yaml
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train_utils.py
│   ├── face_detector.py
│   ├── face_landmarks.py
│   ├── emotion_analyzer.py
│   ├── presentation_metrics.py
│   ├── recommendations.py
│   ├── visualization.py
│   └── report_generator.py
├── scripts/
├── notebooks/
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── outputs/
│   ├── charts/
│   │   ├── exp03_resnet18_finetune_confusion_matrix.png
│   │   ├── exp03_resnet18_finetune_training_curves.png
│   │   ├── exp04_mobilenet_finetune_confusion_matrix.png
│   │   ├── exp04_mobilenet_finetune_training_curves.png
│   │   └── phase3/
│   ├── reports/
│   │   ├── experiments.csv
│   │   └── phase3/
│   ├── screenshots/
│   │   ├── phase1/
│   │   ├── phase2/
│   │   └── phase3/
│   └── videos/
└── docs/
    ├── methodology.md
    ├── experiments.md
    └── limitations.md
```

---

## GitHub and Large Files

This repository should not include large datasets or large model checkpoints.

Ignored by default:

```text
data/*
models/*.pth
models/*.pt
.venv/
outputs/videos/*
temporary root-level reports and charts
```

Curated charts, screenshots, and example reports can be committed for README/demo purposes.

---

## Limitations and Ethics

- PresentSense analyzes visual presentation cues only.
- It does not measure true confidence, personality, mental health, psychological state, or presentation quality absolutely.
- Model-predicted expression cues may be affected by lighting, camera angle, expression ambiguity, occlusions, and FER2013 dataset bias.
- `uncertain` means the expression classifier confidence was below the selected threshold; it is not a negative presentation judgment.
- Looking-forward is an approximation based on face position, Face Mesh nose/eye symmetry, and roll; it is not real eye tracking.
- Head/face stability is based on facial landmark and bounding-box motion; it is not full-body posture analysis.
- The current version does not analyze speech, audio, slide quality, or content.
- Scores and recommendations are heuristic and should be used as supportive feedback, not absolute judgment.
- Videos should be processed locally when possible to protect privacy.


