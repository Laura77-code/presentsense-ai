"""Video analyzer for PresentSense.

Phase 1: OpenCV + MediaPipe face detection.
Phase 2: Optional facial expression recognition when --model is provided.
Phase 3.5: Face Mesh-based visual communication metrics, charts, CSV/JSON,
and Markdown report.

Usage examples:
    python analyze_video.py --source webcam
    python analyze_video.py --source webcam --model models/best_exp03_resnet18_finetune.pth
    python analyze_video.py --source path/to/video.mp4 --model models/best_exp03_resnet18_finetune.pth --output outputs/videos/demo.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import cv2 as cv

from src.emotion_analyzer import EmotionAnalyzer, crop_face_with_margin
from src.face_detector import MediaPipeFaceDetector
from src.face_landmarks import MediaPipeFaceMeshAnalyzer
from src.presentation_metrics import PresentationMetricsTracker
from src.report_generator import generate_markdown_report
from src.visualization import (
    FPSCounter,
    draw_base_overlay,
    draw_expression_overlay,
    draw_face_overlay,
    draw_metric_overlay,
)


def parse_source(source: str) -> Union[int, str]:
    """Convert CLI source into OpenCV-compatible VideoCapture source."""
    if source.lower() == "webcam":
        return 0
    return source


def create_video_writer(output_path: Path, fps: float, frame_width: int, frame_height: int) -> cv.VideoWriter:
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
    smoothing_window: int = 8,
    uncertainty_threshold: float = 0.60,
    generate_report: bool = True,
    reports_dir: Path = Path("outputs/reports"),
    charts_dir: Path = Path("outputs/charts"),
    use_face_mesh: bool = True,
) -> None:
    """Run video analysis with optional expression recognition and Phase 3.5 metrics."""
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
            raise FileNotFoundError(f"Expression model checkpoint not found: {model_path}")
        emotion_analyzer = EmotionAnalyzer(
            model_path,
            smoothing_window=smoothing_window,
            uncertainty_threshold=uncertainty_threshold,
        )
        print(f"Loaded expression model: {model_path}")

    fps_counter = FPSCounter()
    tracker = PresentationMetricsTracker(fps=fps, frame_width=frame_width, frame_height=frame_height)

    print("Starting PresentSense analysis...")
    print("Press ESC or q to stop the preview window.")

    detector_ctx = MediaPipeFaceDetector(min_detection_confidence=min_detection_confidence)
    mesh_ctx: Optional[MediaPipeFaceMeshAnalyzer] = None
    if use_face_mesh:
        mesh_ctx = MediaPipeFaceMeshAnalyzer(min_detection_confidence=min_detection_confidence)

    frame_number = 0
    processed_frames = 0

    try:
        with detector_ctx as detector:
            if mesh_ctx is not None:
                mesh_ctx.__enter__()

            while True:
                valid, frame = capture.read()
                if not valid:
                    break

                frame_number += 1
                if (frame_number - 1) % frame_step != 0:
                    continue

                detection = detector.detect(frame)
                landmark_metrics = mesh_ctx.analyze(frame) if mesh_ctx is not None else None
                current_fps = fps_counter.update()
                prediction = None

                draw_face_overlay(frame, detection)

                if detection is not None and emotion_analyzer is not None:
                    face_crop = crop_face_with_margin(frame, detection.bbox)
                    prediction = emotion_analyzer.predict(face_crop)
                    if prediction is not None:
                        x, y, _, _ = detection.bbox
                        draw_expression_overlay(
                            frame,
                            prediction.display_label,
                            prediction.confidence,
                            top_k=prediction.top_k,
                            origin=(x, y + 25),
                        )

                metric = tracker.update(
                    frame_number=frame_number,
                    detection=detection,
                    prediction=prediction,
                    landmark_metrics=landmark_metrics,
                )
                elapsed = frame_number / fps if fps > 0 else 0.0

                phase_label = "Phase 3.5: Face Mesh Visual Metrics" if generate_report else (
                    "Phase 2: Face + Expression Pipeline" if emotion_analyzer else "Phase 1: Face Pipeline"
                )

                if generate_report:
                    draw_metric_overlay(
                        frame,
                        looking_forward=metric.looking_forward,
                        movement_level=metric.movement_level,
                        elapsed_time=elapsed,
                        yaw_proxy=metric.yaw_proxy,
                        mouth_open_ratio=metric.mouth_open_ratio,
                        face_mesh_detected=metric.face_mesh_detected,
                    )

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
    finally:
        if mesh_ctx is not None:
            mesh_ctx.close()
        capture.release()
        if writer is not None:
            writer.release()
        cv.destroyAllWindows()

    print(f"Done. Processed frames: {processed_frames}")
    if output is not None:
        print(f"Annotated video saved to: {output}")

    if generate_report:
        reports_dir.mkdir(parents=True, exist_ok=True)
        charts_dir.mkdir(parents=True, exist_ok=True)
        summary = tracker.summarize()
        frame_csv = reports_dir / "frame_metrics.csv"
        summary_json = reports_dir / "summary_report.json"
        report_md = reports_dir / "presentation_report.md"
        tracker.save_frame_metrics_csv(frame_csv)
        tracker.save_summary_json(summary_json, summary)
        chart_paths = tracker.save_charts(charts_dir)
        generate_markdown_report(summary, report_md, chart_paths)
        print(f"Frame metrics saved to: {frame_csv}")
        print(f"Summary report saved to: {summary_json}")
        print(f"Markdown report saved to: {report_md}")
        print(f"Charts saved to: {charts_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PresentSense video analyzer")
    parser.add_argument("--source", required=True, help="Use 'webcam' or provide a local video path")
    parser.add_argument("--output", type=Path, default=None, help="Optional output video path, e.g. outputs/videos/demo.mp4")
    parser.add_argument("--model", type=Path, default=None, help="Optional expression model checkpoint path")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window and only save/process video")
    parser.add_argument("--frame-step", type=int, default=1, help="Analyze every Nth frame for faster processing")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum MediaPipe face detection confidence")
    parser.add_argument("--smoothing-window", type=int, default=8, help="Rolling probability window for stable predictions")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.60, help="Show uncertain below this confidence")
    parser.add_argument("--no-report", action="store_true", help="Disable Phase 3.5 CSV/JSON/Markdown report generation")
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"), help="Report output directory")
    parser.add_argument("--charts-dir", type=Path, default=Path("outputs/charts"), help="Chart output directory")
    parser.add_argument("--no-face-mesh", action="store_true", help="Disable Face Mesh metrics and fall back to bounding-box heuristics")
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
        uncertainty_threshold=args.uncertainty_threshold,
        generate_report=not args.no_report,
        reports_dir=args.reports_dir,
        charts_dir=args.charts_dir,
        use_face_mesh=not args.no_face_mesh,
    )


if __name__ == "__main__":
    main()
