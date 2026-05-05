# PresentSense Methodology

## Phase 1: Face Detection Pipeline

The system reads frames from a webcam or local video using OpenCV. Each frame is passed to MediaPipe Face Detection. The highest-confidence face is selected because the target use case is a single presenter practicing in front of a camera.

The video overlay includes:

- Face bounding box.
- Detection confidence.
- FPS.
- Frame number.
- Source label.

## Phase 2: Emotion Recognition Pipeline

Phase 2 adds FER2013 emotion recognition.

### Preprocessing

For each detected face:

1. Crop the face from the frame.
2. Add a margin around the bounding box.
3. Clamp the crop coordinates within the frame boundaries.
4. Convert the image into a PIL image.
5. Convert grayscale/RGB input to 3-channel format.
6. Resize to 224x224.
7. Normalize using ImageNet mean and standard deviation.

### Models

Four models are implemented:

1. Custom CNN from scratch.
2. ResNet18 pretrained with frozen backbone.
3. ResNet18 pretrained with last layers fine-tuned.
4. MobileNetV3 Small pretrained with classifier fine-tuned.

### Temporal Smoothing

Frame-level predictions can flicker. To reduce this, the inference module stores recent probability vectors and averages them over a rolling window. The final label is selected from the averaged probabilities.

### Outputs

Training generates:

- Best model checkpoint.
- Training curves.
- Confusion matrix.
- Metrics CSV.

Video analysis generates:

- Annotated webcam/video preview.
- Optional annotated MP4 output.

## Phase 3 Preview

The next phase will aggregate frame-level results into presentation metrics:

- Emotion distribution.
- Dominant emotion.
- Expressiveness score.
- Looking-forward approximation.
- Head/posture stability.
- Final presentation report.
