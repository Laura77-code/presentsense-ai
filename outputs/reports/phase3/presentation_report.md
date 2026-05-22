# Presentation Visual Feedback Report

> PresentSense provides heuristic visual communication feedback. It is not a medical, psychological, or clinical diagnosis tool.

## 1. Video Information

- Duration: **19.2 seconds**
- FPS: **30.0**
- Analyzed frames: **576**
- Face detected frames: **576**
- Face detection rate: **100.0%**
- Face Mesh detected frames: **576**
- Face Mesh detection rate: **100.0%**

## 2. Visible Expression Summary

- Dominant model-predicted visible cue: **uncertain**
- Average model confidence: **0.6727**

> The expression labels are model-predicted visual cues. Frames below the confidence threshold are reported as `uncertain` instead of forcing a label.

### Expression Distribution

- happy: **47.05%**
- sad: **0.17%**
- surprise: **4.34%**
- uncertain: **48.44%**

## 3. Geometric Face-Mesh Cues

- Mean absolute yaw proxy: **0.1066**
- Mean absolute roll angle: **4.1706 degrees**
- Mean mouth openness ratio: **0.0125**
- Mouth openness variation: **0.027**

> These are geometric landmark cues. The yaw value is a normalized proxy, not a calibrated 3D head-pose angle.

## 4. Visual Communication Scores

- Looking-forward approximation: **84.9/100**
- Visible expression variation: **55.8/100**
- Head/face stability: **83.76/100**
- Expression variety: **39.45/100**
- Final overall visual score: **71.34/100**

## 5. Timeline Highlights

- Strongest expression segment: **8.13s**
- Strongest model-predicted cue: **happy**
- Low-confidence segment: **0.73s**
- Highest head/face movement segment: **7.5s**

## 6. Generated Charts

- expression_timeline: `outputs\charts\phase3_expression_timeline.png`
- expression_distribution: `outputs\charts\phase3_expression_distribution.png`
- looking_forward: `outputs\charts\phase3_looking_forward_over_time.png`
- head_face_movement: `outputs\charts\phase3_head_face_movement_over_time.png`
- mouth_openness: `outputs\charts\phase3_mouth_openness_over_time.png`

## 7. Recommendations

- Good looking-forward consistency. Your face stayed visually oriented toward the camera/audience for much of the recording.
- Many frames were classified as uncertain. Improve front lighting, camera angle, and face visibility before interpreting expression cues strongly.
- Good head/face stability. Your face position was controlled enough for a clear camera presentation.
- Positive/warm model-predicted cues appeared during the recording, which can help create a friendly presentation tone.
- Overall visual delivery looks strong. Continue practicing with consistent lighting and a camera at eye level.

## 8. Limitations

- This report analyzes visual cues only.
- It does not measure real emotion, confidence, personality, mental health, or presentation quality absolutely.
- Model-predicted expression cues may be affected by lighting, camera angle, expression ambiguity, occlusions, and FER2013 dataset bias.
- `uncertain` means the expression classifier confidence was below the selected threshold; it is not a negative presentation judgment.
- Looking-forward is an approximation based on face position, Face Mesh nose/eye symmetry, and roll; it is not real eye tracking.
- Head/face stability is based on facial landmark and bounding-box motion; it is not full-body posture analysis.
- Scores and recommendations are heuristic and should be used only as supportive practice feedback.
