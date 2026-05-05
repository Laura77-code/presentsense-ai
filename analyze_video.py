"""Video analyzer for PresentSense Phase 1/2.

Phase 1: OpenCV + MediaPipe face detection.
Phase 2: Optional facial emotion recognition when --model is provided.

Usage examples:
    python analyze_video.py --source webcam
    python analyze_video.py --source webcam --model models/best_emotion_model.pth
    python analyze_video.py --source path/to/video.mp4 --model models/best_emotion_model.pth --output outputs/videos/demo_phase2.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import cv2 as cv

from src.emotion_analyzer import EmotionAnalyzer, crop_face_with_margin
from src.face_detector import MediaPipeFaceDetector
from src.visualization import FPSCounter, draw_base_overlay, draw_emotion_overlay, draw_face_overlay


def parse_source(source: str) -> Union[int, str]:
    """Convert CLI source into OpenCV-compatible VideoCapture source."""
    if source.lower() == "webcam":
        return 0
    return source


def create_video_writer(
    output_path: Path,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> cv.VideoWriter:
    """Create an OpenCV VideoWriter for MP4 output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    return cv.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))


def open_capture(source: Union[int, str]) -> cv.VideoCapture:
    """Open webcam or local video and raise a helpful error if it fails."""
    capture = cv.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source}. "
            "Check that your webcam is available or that the video path exists."
        )
    return capture


def analyze_video(
    source: str,
    output: Optional[Path] = None,
    display: bool = True,
    frame_step: int = 1,
    min_detection_confidence: float = 0.5,
    model_path: Optional[Path] = None,
    smoothing_window: int = 5,
) -> None:
    """Run the face-detection pipeline and optionally add emotion recognition."""
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")

    cv_source = parse_source(source)
    source_label = "webcam" if source.lower() == "webcam" else str(source)
    capture = open_capture(cv_source)

    input_fps = capture.get(cv.CAP_PROP_FPS)
    fps = input_fps if input_fps and input_fps > 1 else 30.0
    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    writer: Optional[cv.VideoWriter] = None
    if output is not None:
        writer = create_video_writer(output, fps, frame_width, frame_height)
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video writer: {output}")

    emotion_analyzer: Optional[EmotionAnalyzer] = None
    if model_path is not None:
        if not model_path.exists():
            capture.release()
            raise FileNotFoundError(f"Emotion model checkpoint not found: {model_path}")
        emotion_analyzer = EmotionAnalyzer(model_path, smoothing_window=smoothing_window)
        print(f"Loaded emotion model: {model_path}")

    fps_counter = FPSCounter()

    print("Starting PresentSense analysis...")
    print("Press ESC or q to stop the preview window.")

    with MediaPipeFaceDetector(min_detection_confidence=min_detection_confidence) as detector:
        frame_number = 0
        processed_frames = 0

        while True:
            valid, frame = capture.read()
            if not valid:
                break

            frame_number += 1
            if (frame_number - 1) % frame_step != 0:
                continue

            detection = detector.detect(frame)
            current_fps = fps_counter.update()

            draw_face_overlay(frame, detection)

            if detection is not None and emotion_analyzer is not None:
                face_crop = crop_face_with_margin(frame, detection.bbox)
                prediction = emotion_analyzer.predict(face_crop)
                if prediction is not None:
                    x, y, _, _ = detection.bbox
                    draw_emotion_overlay(
                        frame,
                        prediction.label,
                        prediction.confidence,
                        origin=(x, y + 25),
                    )

            phase_label = "Phase 2: Face + Emotion Pipeline" if emotion_analyzer else "Phase 1: Face Pipeline"
            draw_base_overlay(
                frame,
                frame_number=frame_number,
                fps=current_fps,
                source_label=source_label,
                phase_label=phase_label,
            )

            if writer is not None:
                writer.write(frame)

            if display:
                cv.imshow("PresentSense", frame)
                key = cv.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            processed_frames += 1

    capture.release()
    if writer is not None:
        writer.release()
    cv.destroyAllWindows()

    print(f"Done. Processed frames: {processed_frames}")
    if output is not None:
        print(f"Annotated video saved to: {output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PresentSense Phase 1/2 video analyzer")
    parser.add_argument("--source", required=True, help="Use 'webcam' or provide a local video path")
    parser.add_argument("--output", type=Path, default=None, help="Optional output video path, e.g. outputs/videos/demo_phase2.mp4")
    parser.add_argument("--model", type=Path, default=None, help="Optional emotion model checkpoint path")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window and only save/process video")
    parser.add_argument("--frame-step", type=int, default=1, help="Analyze every Nth frame for faster processing")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum MediaPipe face detection confidence")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Rolling probability window for stable emotion predictions")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    analyze_video(
        source=args.source,
        output=args.output,
        display=not args.no_display,
        frame_step=args.frame_step,
        min_detection_confidence=args.min_confidence,
        model_path=args.model,
        smoothing_window=args.smoothing_window,
    )


if __name__ == "__main__":
    main()
