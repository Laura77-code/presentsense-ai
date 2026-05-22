"""Markdown report generation for PresentSense Phase 3.5."""

from __future__ import annotations

from pathlib import Path

from src.recommendations import generate_recommendations


def generate_markdown_report(summary: dict, output_path: str | Path, chart_paths: dict[str, str] | None = None) -> Path:
    """Generate a human-readable presentation visual feedback report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_paths = chart_paths or {}
    recommendations = generate_recommendations(summary)

    video = summary.get("video_info", {})
    expression = summary.get("expression_summary", summary.get("emotion_summary", {}))
    percentages = summary.get("expression_percentages", summary.get("emotion_percentages", {}))
    scores = summary.get("scores", {})
    highlights = summary.get("timeline_highlights", {})
    geometry = summary.get("head_face_geometry", {})

    lines: list[str] = []
    lines.append("# Presentation Visual Feedback Report")
    lines.append("")
    lines.append("> PresentSense provides heuristic visual communication feedback. It is not a medical, psychological, or clinical diagnosis tool.")
    lines.append("")

    lines.append("## 1. Video Information")
    lines.append("")
    lines.append(f"- Duration: **{video.get('duration_sec', 0)} seconds**")
    lines.append(f"- FPS: **{video.get('fps', 0)}**")
    lines.append(f"- Analyzed frames: **{video.get('analyzed_frames', 0)}**")
    lines.append(f"- Face detected frames: **{video.get('face_detected_frames', 0)}**")
    lines.append(f"- Face detection rate: **{video.get('face_detection_rate', 0)}%**")
    lines.append(f"- Face Mesh detected frames: **{video.get('face_mesh_detected_frames', 0)}**")
    lines.append(f"- Face Mesh detection rate: **{video.get('face_mesh_detection_rate', 0)}%**")
    lines.append("")

    lines.append("## 2. Visible Expression Summary")
    lines.append("")
    lines.append(f"- Dominant model-predicted visible cue: **{expression.get('dominant_expression', 'not_available')}**")
    lines.append(f"- Average model confidence: **{expression.get('average_confidence', 0)}**")
    lines.append("")
    lines.append("> The expression labels are model-predicted visual cues. Frames below the confidence threshold are reported as `uncertain` instead of forcing a label.")
    lines.append("")

    if percentages:
        lines.append("### Expression Distribution")
        lines.append("")
        for label, pct in percentages.items():
            lines.append(f"- {label}: **{pct}%**")
        lines.append("")

    lines.append("## 3. Geometric Face-Mesh Cues")
    lines.append("")
    lines.append(f"- Mean absolute yaw proxy: **{geometry.get('mean_abs_yaw_proxy', 0)}**")
    lines.append(f"- Mean absolute roll angle: **{geometry.get('mean_abs_roll_deg', 0)} degrees**")
    lines.append(f"- Mean mouth openness ratio: **{geometry.get('mean_mouth_open_ratio', 0)}**")
    lines.append(f"- Mouth openness variation: **{geometry.get('mouth_open_variation', 0)}**")
    lines.append("")
    lines.append("> These are geometric landmark cues. The yaw value is a normalized proxy, not a calibrated 3D head-pose angle.")
    lines.append("")

    lines.append("## 4. Visual Communication Scores")
    lines.append("")
    lines.append(f"- Looking-forward approximation: **{scores.get('looking_forward_score', 0)}/100**")
    lines.append(f"- Visible expression variation: **{scores.get('visible_expression_variation_score', scores.get('expressiveness_score', 0))}/100**")
    lines.append(f"- Head/face stability: **{scores.get('head_face_stability_score', scores.get('posture_stability_score', 0))}/100**")
    lines.append(f"- Expression variety: **{scores.get('expression_variety_score', scores.get('emotion_balance_score', 0))}/100**")
    lines.append(f"- Final overall visual score: **{scores.get('overall_score', 0)}/100**")
    lines.append("")

    lines.append("## 5. Timeline Highlights")
    lines.append("")
    lines.append(f"- Strongest expression segment: **{highlights.get('strongest_expression_segment_sec', 'N/A')}s**")
    lines.append(f"- Strongest model-predicted cue: **{highlights.get('strongest_expression', 'N/A')}**")
    lines.append(f"- Low-confidence segment: **{highlights.get('lowest_confidence_segment_sec', 'N/A')}s**")
    lines.append(f"- Highest head/face movement segment: **{highlights.get('highest_head_face_movement_segment_sec', highlights.get('unstable_movement_segment_sec', 'N/A'))}s**")
    lines.append("")

    if chart_paths:
        lines.append("## 6. Generated Charts")
        lines.append("")
        for name, path in chart_paths.items():
            lines.append(f"- {name}: `{path}`")
        lines.append("")

    lines.append("## 7. Recommendations")
    lines.append("")
    for rec in recommendations:
        lines.append(f"- {rec}")
    lines.append("")

    lines.append("## 8. Limitations")
    lines.append("")
    lines.append("- This report analyzes visual cues only.")
    lines.append("- It does not measure real emotion, confidence, personality, mental health, or presentation quality absolutely.")
    lines.append("- Model-predicted expression cues may be affected by lighting, camera angle, expression ambiguity, occlusions, and FER2013 dataset bias.")
    lines.append("- `uncertain` means the expression classifier confidence was below the selected threshold; it is not a negative presentation judgment.")
    lines.append("- Looking-forward is an approximation based on face position, Face Mesh nose/eye symmetry, and roll; it is not real eye tracking.")
    lines.append("- Head/face stability is based on facial landmark and bounding-box motion; it is not full-body posture analysis.")
    lines.append("- Scores and recommendations are heuristic and should be used only as supportive practice feedback.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
