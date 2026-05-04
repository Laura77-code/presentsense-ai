# PresentSense: A Computer Vision-Based Presentation Coach

## Overview

PresentSense is a Computer Vision final project that analyzes a presentation video or webcam stream and provides visual feedback about how a student presents in front of a camera.

This repository is being built in four phases. The current implementation covers **Phase 1 only**: a working OpenCV + MediaPipe video pipeline with face detection, overlays, webcam input, local video input, and optional annotated video export.

> Important: PresentSense is not a medical, psychological, or clinical diagnosis tool. It is only a visual communication feedback tool for presentation practice.

## Motivation

Students often practice presentations alone and do not receive feedback about visual communication cues such as whether they are visible, facing the audience, or moving too much. PresentSense aims to provide a practical computer-vision-based assistant that can later summarize visual presentation behavior and provide improvement recommendations.

## Computer Vision Concepts Used

Phase 1 already uses:

- OpenCV video input and output.
- Frame-by-frame video processing.
- MediaPipe face detection.
- OpenCV drawing functions and overlays.
- FPS and frame metadata visualization.
- Robust handling of missing faces.

Later phases will add:

- Facial emotion recognition.
- Transfer learning for image classification.
- Face crop preprocessing.
- Temporal smoothing and aggregation.
- Presentation metrics and reports.
- Streamlit interface.

## Features

### Implemented in Phase 1

- Webcam analysis.
- Local video analysis.
- Face detection using MediaPipe.
- Bounding box overlay.
- Face detection confidence overlay.
- FPS overlay.
- Frame number overlay.
- Status text when no face is detected.
- Annotated video export to `outputs/videos/`.

### Planned for Later Phases

- FER2013 emotion classification.
- Emotion timeline and distribution.
- Looking-forward score.
- Facial expressiveness score.
- Head movement and posture stability score.
- Optional gesture activity score.
- Markdown/JSON/CSV reports.
- Streamlit app.

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
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train_utils.py
│   ├── face_detector.py
│   ├── emotion_analyzer.py
│   ├── presentation_metrics.py
│   ├── recommendations.py
│   ├── visualization.py
│   └── report_generator.py
│
├── scripts/
│   ├── prepare_fer2013.py
│   ├── run_webcam_demo.py
│   ├── run_video_demo.py
│   └── export_demo_assets.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── outputs/
│   ├── .gitkeep
│   ├── videos/
│   ├── reports/
│   ├── charts/
│   └── screenshots/
│
└── docs/
    ├── methodology.md
    ├── experiments.md
    └── limitations.md
```

## Installation

Create and activate a virtual environment.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running Phase 1

### Webcam Demo

```bash
python analyze_video.py --source webcam
```

Press `q` or `ESC` to stop the preview window.

### Webcam Demo and Save Video

```bash
python analyze_video.py --source webcam --output outputs/videos/webcam_phase1.mp4
```

### Local Video Demo

```bash
python analyze_video.py --source path/to/video.mp4 --output outputs/videos/demo_phase1.mp4
```

### Faster Processing

Analyze every 3rd frame:

```bash
python analyze_video.py --source path/to/video.mp4 --output outputs/videos/demo_phase1.mp4 --frame-step 3
```

### Headless Processing

Useful when running in an environment without GUI support:

```bash
python analyze_video.py --source path/to/video.mp4 --output outputs/videos/demo_phase1.mp4 --no-display
```

## Phase 1 Acceptance Criteria

- [ ] Webcam opens successfully.
- [ ] A local video can be analyzed.
- [ ] A face bounding box is drawn when a face is detected.
- [ ] `No face detected` is shown when no face is visible.
- [ ] FPS and frame number are displayed.
- [ ] Annotated video can be saved to `outputs/videos/`.
- [ ] The project runs from the repository root.

## Dataset

No dataset is required for Phase 1.

FER2013 will be added in Phase 2. The planned dataset location will be:

```text
data/fer2013/fer2013.csv
```

or:

```text
data/fer2013/train/<class_name>/
data/fer2013/test/<class_name>/
```

## Training

Training starts in Phase 2. For now, `train.py` is a placeholder.

## Methodology

The Phase 1 pipeline follows this process:

1. Open webcam or local video with OpenCV.
2. Read frames one by one.
3. Convert each frame to RGB for MediaPipe.
4. Detect the presenter's face.
5. Convert relative MediaPipe bounding box coordinates to pixel coordinates.
6. Draw overlays using OpenCV.
7. Save annotated video if an output path is provided.

## Limitations

- Phase 1 only detects faces.
- No emotion recognition is implemented yet.
- No eye contact, posture, gesture, or scoring metrics are implemented yet.
- Face detection may fail with poor lighting, extreme head angles, occlusions, or low camera quality.
- This project must not be interpreted as measuring real confidence, personality, mental health, or psychological state.

## Future Work

- Add FER2013 emotion classification.
- Train and compare CNN, ResNet18, and MobileNetV3 models.
- Add temporal smoothing for emotion predictions.
- Add visual presentation metrics.
- Generate charts and reports.
- Build a Streamlit app.
- Add optional audio/speech feedback in a future extension.

## License

MIT License.
