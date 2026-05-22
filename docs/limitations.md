# Limitations and Ethics

PresentSense is a computer vision project for presentation practice. It must not be used as a medical, psychological, clinical, hiring, grading, or personality-assessment tool.

## Visual-Only Analysis

The system analyzes visual cues from webcam or video frames only. It does not analyze speech content, voice, slides, or audience response.

## Expression Recognition Limitations

The expression classifier is trained on FER2013, which contains small and noisy facial expression images. Real webcam input may differ from the dataset because of lighting, camera angle, face scale, background, occlusion, and individual expression style. Some classes such as fear, disgust, anger, and surprise can be difficult to classify reliably.

## Looking-Forward Is Approximate

The looking-forward metric is not real gaze tracking. It uses face position and Face Mesh geometry such as nose/eye symmetry and roll angle. A person can look away with their eyes while their face remains frontal, so this metric should be interpreted only as a frontal-face approximation.

## Head/Face Stability Is Not Full-Body Posture

The stability score measures head/face movement in the camera frame. It does not measure spine posture, shoulders, body alignment, or ergonomics.

## Heuristic Scores

The final scores are heuristic communication-practice metrics. They are useful for observing patterns and generating feedback, but they are not absolute measurements of confidence, emotion, engagement, or presentation quality.

## Privacy

Presentation videos may contain sensitive personal information. The recommended use is local processing whenever possible. Datasets, private videos, and large model files should not be committed to GitHub.
