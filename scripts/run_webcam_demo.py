"""Convenience script for running the Phase 1 webcam demo."""

from analyze_video import analyze_video

if __name__ == "__main__":
    analyze_video(source="webcam", output=None, display=True)
