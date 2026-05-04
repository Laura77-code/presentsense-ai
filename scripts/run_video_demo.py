"""Convenience script for running the Phase 1 local video demo."""

from pathlib import Path

from analyze_video import analyze_video

if __name__ == "__main__":
    analyze_video(
        source="data/samples/presentation.mp4",
        output=Path("outputs/videos/demo_phase1.mp4"),
        display=True,
    )
