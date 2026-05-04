"""Phase 1 video analyzer for PresentSense.

Usage examples:
    python analyze_video.py --source webcam
    python analyze_video.py --source path/to/video.mp4 --output outputs/videos/demo_phase1.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import cv2 as cv

from src.face_detector import MediaPipeFaceDetector
from src.visualization import FPSCounter, draw_base_overlay, draw_face_overlay


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
) -> None:
    """Run the Phase 1 face-detection pipeline on webcam or video."""
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

    fps_counter = FPSCounter()

    print("Starting PresentSense Phase 1 analysis...")
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
            draw_base_overlay(frame, frame_number=frame_number, fps=current_fps, source_label=source_label)

            if writer is not None:
                writer.write(frame)

            if display:
                cv.imshow("PresentSense Phase 1", frame)
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
    parser = argparse.ArgumentParser(description="PresentSense Phase 1: OpenCV + MediaPipe face pipeline")
    parser.add_argument("--source", required=True, help="Use 'webcam' or provide a local video path")
    parser.add_argument("--output", type=Path, default=None, help="Optional output video path, e.g. outputs/videos/demo_phase1.mp4")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window and only save/process video")
    parser.add_argument("--frame-step", type=int, default=1, help="Analyze every Nth frame for faster processing")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum MediaPipe face detection confidence")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    analyze_video(
        source=args.source,
        output=args.output,
        display=not args.no_display,
        frame_step=args.frame_step,
        min_detection_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
