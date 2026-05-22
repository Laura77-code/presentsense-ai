# Methodology

## Phase 1: Video and Face Detection Pipeline

PresentSense reads webcam or video frames using OpenCV `VideoCapture`, processes frames sequentially, and overlays results using OpenCV drawing functions. MediaPipe Face Detection is used to locate the presenter's face. If an output path is provided, annotated frames are written with OpenCV `VideoWriter`.

## Phase 2: Visible Facial Expression Recognition

The expression classifier is trained on FER2013 using transfer learning. The main demo model is a ResNet18 fine-tuned model. During inference, the detected face is cropped with margin, resized to 224 x 224, normalized with ImageNet statistics, and passed through the PyTorch model. The system displays a visible facial expression cue and confidence, not a psychological interpretation.

## Phase 3.5: Presentation Visual Metrics

Phase 3.5 uses a hybrid design:

1. A trained expression classifier estimates visible facial expression cues.
2. MediaPipe Face Mesh landmarks provide simple geometric face measurements.
3. Temporal aggregation converts frame-level signals into report-level scores.

The metrics are heuristic and intended for presentation practice only.

### Looking-Forward Approximation

This is not eye tracking. It estimates whether the presenter is visually oriented toward the camera using:

- detected face position in the frame,
- face size,
- nose position relative to the eye midpoint,
- approximate face roll from the eye line.

A frame is counted as looking-forward when the face is centered, reasonably sized, has limited roll, and the nose is close to the midpoint between both eyes.

### Head/Face Stability

This is not full-body posture analysis. It measures the stability of the face/head in the camera frame using:

- movement of the face bounding-box center,
- movement of the Face Mesh nose landmark.

Excessive motion lowers the score, while small natural movement is allowed.

### Visible Expression Variation

This is not a measure of true emotion or personality. It estimates whether the visible facial cues vary over time using:

- expression class diversity,
- variation in expression probabilities,
- mouth openness variation from Face Mesh,
- eye openness variation from Face Mesh.

### Expression Variety

Expression variety measures whether a single expression cue dominates the whole video. A low score means that one predicted visible cue appears for most frames. This can also reflect model bias or difficult lighting conditions, so it should be interpreted carefully.

### Overall Visual Score

The overall score uses a weighted heuristic formula:

```text
overall_score =
    0.35 * looking_forward_score
  + 0.30 * visible_expression_variation_score
  + 0.25 * head_face_stability_score
  + 0.10 * expression_variety_score
```

The final score is supportive feedback only, not an objective grade of presentation ability.
