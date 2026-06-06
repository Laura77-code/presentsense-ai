# Methodology and Limitations

## Face Detection

MediaPipe detects the presenter's face in each frame. OpenCV draws the bounding box, FPS, and status overlays.

## Expression Recognition

1. The detected face is cropped with a margin.
2. The crop is resized to `224 x 224`.
3. The crop is normalized using ImageNet statistics.
4. A PyTorch model predicts FER2013 expression classes.
5. Softmax probabilities are smoothed over time.
6. If confidence is below the threshold, the frame is labeled as `uncertain`.

## Face Mesh Geometry

Face Mesh landmarks are used to estimate geometric presentation cues:

- Nose and eye symmetry for looking-forward approximation.
- Face roll angle proxy.
- Nose/landmark movement for head/face stability.
- Mouth openness ratio over time.
- Eye openness variation as part of expression variation.

## Scores

The final score is a heuristic weighted score:

```text
overall_score =
  0.35 * looking_forward_score
+ 0.30 * visible_expression_variation_score
+ 0.25 * head_face_stability_score
+ 0.10 * expression_variety_score
```

## Limitations

- PresentSense analyzes visual cues only.
- It does not measure true emotion, confidence, personality, mental health, or absolute presentation quality.
- Expression predictions can be affected by lighting, camera angle, occlusion, and FER2013 dataset bias.
- `uncertain` means model confidence was below the selected threshold; it is not a negative judgment.
- Looking-forward is an approximation, not real gaze tracking or eye tracking.
- Head/face stability is not full-body posture analysis.
- Audio, speech quality, slide content, and presentation structure are not analyzed.
