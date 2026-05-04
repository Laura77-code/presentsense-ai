# Methodology

## Phase 1: Video and Face Detection Pipeline

PresentSense starts with a robust video processing pipeline using OpenCV and MediaPipe Face Detection.

Current pipeline:

1. Open a webcam stream or local video file with `cv.VideoCapture`.
2. Read frames sequentially.
3. Convert each frame from BGR to RGB for MediaPipe.
4. Detect the presenter's face.
5. Draw a bounding box, confidence label, FPS, frame number, and status overlay.
6. Optionally save the annotated output using `cv.VideoWriter`.

Later phases will add emotion recognition, temporal aggregation, presentation scores, charts, reports, and Streamlit UI.
