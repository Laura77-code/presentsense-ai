"""Presentation visual metrics for PresentSense Phase 3.5.

The scores in this module are heuristic communication-practice metrics. They are
not medical, psychological, clinical, or full-body posture measurements.

Phase 3.5 improves the previous bounding-box-only metrics by optionally using
MediaPipe Face Mesh landmarks:
- looking-forward approximation: face center + nose/eye symmetry + roll proxy
- head/face stability: nose/landmark movement over time
- visible expression variation: expression probability diversity + mouth/eye motion
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.emotion_analyzer import EmotionPrediction
from src.face_detector import FaceDetectionResult
from src.face_landmarks import FaceLandmarkMetrics


@dataclass
class FrameMetric:
    frame_number: int
    timestamp_sec: float
    face_detected: bool
    face_confidence: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    center_x_norm: float
    center_y_norm: float
    bbox_area_ratio: float
    face_mesh_detected: bool
    landmark_center_x_norm: float
    landmark_center_y_norm: float
    nose_x_norm: float
    nose_y_norm: float
    yaw_proxy: float
    roll_deg: float
    mouth_open_ratio: float
    eye_open_ratio: float
    looking_forward: Optional[bool]
    movement_magnitude: float
    landmark_movement_magnitude: float
    movement_level: str
    expression: str
    expression_confidence: float
    raw_expression: str
    top1: str
    top1_confidence: float
    top2: str
    top2_confidence: float
    top3: str
    top3_confidence: float


@dataclass
class PresentationMetricsTracker:
    """Accumulate per-frame CV metrics and summarize a presentation video."""

    fps: float
    frame_width: int
    frame_height: int
    previous_center: Optional[tuple[float, float]] = None
    previous_nose: Optional[tuple[float, float]] = None
    records: list[FrameMetric] = field(default_factory=list)

    def update(
        self,
        frame_number: int,
        detection: Optional[FaceDetectionResult],
        prediction: Optional[EmotionPrediction],
        landmark_metrics: Optional[FaceLandmarkMetrics] = None,
    ) -> FrameMetric:
        timestamp = frame_number / self.fps if self.fps > 0 else 0.0

        if detection is None:
            metric = FrameMetric(
                frame_number=frame_number,
                timestamp_sec=timestamp,
                face_detected=False,
                face_confidence=0.0,
                bbox_x=0,
                bbox_y=0,
                bbox_w=0,
                bbox_h=0,
                center_x_norm=0.0,
                center_y_norm=0.0,
                bbox_area_ratio=0.0,
                face_mesh_detected=False,
                landmark_center_x_norm=0.0,
                landmark_center_y_norm=0.0,
                nose_x_norm=0.0,
                nose_y_norm=0.0,
                yaw_proxy=0.0,
                roll_deg=0.0,
                mouth_open_ratio=0.0,
                eye_open_ratio=0.0,
                looking_forward=None,
                movement_magnitude=0.0,
                landmark_movement_magnitude=0.0,
                movement_level="no_face",
                expression="no_face",
                expression_confidence=0.0,
                raw_expression="no_face",
                top1="",
                top1_confidence=0.0,
                top2="",
                top2_confidence=0.0,
                top3="",
                top3_confidence=0.0,
            )
            self.records.append(metric)
            return metric

        x, y, w, h = detection.bbox
        cx = (x + w / 2) / max(1, self.frame_width)
        cy = (y + h / 2) / max(1, self.frame_height)
        area_ratio = (w * h) / max(1, self.frame_width * self.frame_height)

        if self.previous_center is None:
            bbox_movement = 0.0
        else:
            px, py = self.previous_center
            bbox_movement = float(np.sqrt((cx - px) ** 2 + (cy - py) ** 2))
        self.previous_center = (cx, cy)

        face_mesh_detected = landmark_metrics is not None and landmark_metrics.detected
        landmark_movement = 0.0
        yaw_proxy = 0.0
        roll_deg = 0.0
        mouth_open_ratio = 0.0
        eye_open_ratio = 0.0
        nose_x_norm = 0.0
        nose_y_norm = 0.0
        landmark_cx = 0.0
        landmark_cy = 0.0

        if face_mesh_detected and landmark_metrics is not None:
            nose_x_norm = landmark_metrics.nose_x_norm
            nose_y_norm = landmark_metrics.nose_y_norm
            landmark_cx = landmark_metrics.face_center_x_norm
            landmark_cy = landmark_metrics.face_center_y_norm
            yaw_proxy = landmark_metrics.yaw_proxy
            roll_deg = landmark_metrics.roll_deg
            mouth_open_ratio = landmark_metrics.mouth_open_ratio
            eye_open_ratio = landmark_metrics.eye_open_ratio

            if self.previous_nose is None:
                landmark_movement = 0.0
            else:
                px, py = self.previous_nose
                landmark_movement = float(np.sqrt((nose_x_norm - px) ** 2 + (nose_y_norm - py) ** 2))
            self.previous_nose = (nose_x_norm, nose_y_norm)

            looking_forward = estimate_looking_forward_with_landmarks(
                bbox_center=(cx, cy),
                area_ratio=area_ratio,
                yaw_proxy=yaw_proxy,
                roll_deg=roll_deg,
            )
            movement = 0.65 * landmark_movement + 0.35 * bbox_movement
        else:
            looking_forward = estimate_looking_forward_with_bbox(cx, cy, area_ratio)
            movement = bbox_movement

        movement_level = classify_movement(movement)

        if prediction is None:
            display_expression = "not_available"
            expression_confidence = 0.0
            raw_expression = "not_available"
            top = [("", 0.0), ("", 0.0), ("", 0.0)]
        else:
            display_expression = prediction.display_label
            expression_confidence = prediction.confidence
            raw_expression = prediction.label
            top = (prediction.top_k + [("", 0.0), ("", 0.0), ("", 0.0)])[:3]

        metric = FrameMetric(
            frame_number=frame_number,
            timestamp_sec=timestamp,
            face_detected=True,
            face_confidence=float(detection.confidence),
            bbox_x=x,
            bbox_y=y,
            bbox_w=w,
            bbox_h=h,
            center_x_norm=float(cx),
            center_y_norm=float(cy),
            bbox_area_ratio=float(area_ratio),
            face_mesh_detected=face_mesh_detected,
            landmark_center_x_norm=float(landmark_cx),
            landmark_center_y_norm=float(landmark_cy),
            nose_x_norm=float(nose_x_norm),
            nose_y_norm=float(nose_y_norm),
            yaw_proxy=float(yaw_proxy),
            roll_deg=float(roll_deg),
            mouth_open_ratio=float(mouth_open_ratio),
            eye_open_ratio=float(eye_open_ratio),
            looking_forward=looking_forward,
            movement_magnitude=float(movement),
            landmark_movement_magnitude=float(landmark_movement),
            movement_level=movement_level,
            expression=display_expression,
            expression_confidence=float(expression_confidence),
            raw_expression=raw_expression,
            top1=top[0][0],
            top1_confidence=float(top[0][1]),
            top2=top[1][0],
            top2_confidence=float(top[1][1]),
            top3=top[2][0],
            top3_confidence=float(top[2][1]),
        )
        self.records.append(metric)
        return metric

    def summarize(self) -> dict:
        total_frames = len(self.records)
        if total_frames == 0:
            return empty_summary()

        face_frames = [r for r in self.records if r.face_detected]
        mesh_frames = [r for r in face_frames if r.face_mesh_detected]
        # Use the displayed label for report statistics. This intentionally keeps
        # "uncertain" in the distribution so the report does not overstate
        # confidence when the classifier is unsure.
        expression_frames = [r for r in self.records if r.expression not in {"no_face", "not_available"}]
        confident_expression_frames = [r for r in expression_frames if r.expression != "uncertain"]
        detected_count = len(face_frames)
        duration = max((r.timestamp_sec for r in self.records), default=0.0)

        expression_counts = Counter(r.expression for r in expression_frames)
        confident_expression_counts = Counter(r.expression for r in confident_expression_frames)
        expression_percentages = {
            label: round(100.0 * count / max(1, len(expression_frames)), 2)
            for label, count in sorted(expression_counts.items())
        }
        dominant_expression = expression_counts.most_common(1)[0][0] if expression_counts else "not_available"

        confidences = [r.expression_confidence for r in expression_frames]
        movements = [r.movement_magnitude for r in face_frames if r.movement_level != "no_face"]
        landmark_movements = [r.landmark_movement_magnitude for r in mesh_frames]
        looking_values = [r.looking_forward for r in face_frames if r.looking_forward is not None]
        mouth_values = [r.mouth_open_ratio for r in mesh_frames if r.mouth_open_ratio > 0]
        eye_values = [r.eye_open_ratio for r in mesh_frames if r.eye_open_ratio > 0]
        yaw_values = [r.yaw_proxy for r in mesh_frames]
        roll_values = [r.roll_deg for r in mesh_frames]

        looking_score = 100.0 * sum(bool(v) for v in looking_values) / max(1, len(looking_values))
        stability_score = compute_head_face_stability_score(movements)
        expression_variety_score = compute_expression_variety_score(expression_percentages)
        visible_expression_variation_score = compute_visible_expression_variation_score(
            expression_percentages=expression_percentages,
            counts=confident_expression_counts,
            confidences=confidences,
            mouth_values=mouth_values,
            eye_values=eye_values,
        )
        overall = (
            0.35 * looking_score
            + 0.30 * visible_expression_variation_score
            + 0.25 * stability_score
            + 0.10 * expression_variety_score
        )

        return {
            "video_info": {
                "duration_sec": round(duration, 2),
                "fps": round(self.fps, 2),
                "analyzed_frames": total_frames,
                "face_detected_frames": detected_count,
                "face_detection_rate": round(100.0 * detected_count / max(1, total_frames), 2),
                "face_mesh_detected_frames": len(mesh_frames),
                "face_mesh_detection_rate": round(100.0 * len(mesh_frames) / max(1, total_frames), 2),
            },
            "expression_summary": {
                "dominant_expression": dominant_expression,
                "average_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 4),
            },
            # Backward-compatible key for older README/report code.
            "emotion_summary": {
                "dominant_expression": dominant_expression,
                "average_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 4),
            },
            "expression_percentages": expression_percentages,
            "emotion_percentages": expression_percentages,
            "head_face_geometry": {
                "mean_yaw_proxy": round(float(np.mean(yaw_values)) if yaw_values else 0.0, 4),
                "mean_abs_yaw_proxy": round(float(np.mean(np.abs(yaw_values))) if yaw_values else 0.0, 4),
                "mean_roll_deg": round(float(np.mean(roll_values)) if roll_values else 0.0, 4),
                "mean_abs_roll_deg": round(float(np.mean(np.abs(roll_values))) if roll_values else 0.0, 4),
                "mean_mouth_open_ratio": round(float(np.mean(mouth_values)) if mouth_values else 0.0, 4),
                "mouth_open_variation": round(float(np.std(mouth_values)) if mouth_values else 0.0, 4),
                "eye_open_variation": round(float(np.std(eye_values)) if eye_values else 0.0, 4),
            },
            "movement": {
                "mean_movement": round(float(np.mean(movements)) if movements else 0.0, 6),
                "max_movement": round(float(np.max(movements)) if movements else 0.0, 6),
                "mean_landmark_movement": round(float(np.mean(landmark_movements)) if landmark_movements else 0.0, 6),
                "max_landmark_movement": round(float(np.max(landmark_movements)) if landmark_movements else 0.0, 6),
            },
            "scores": {
                "looking_forward_score": round(looking_score, 2),
                "visible_expression_variation_score": round(visible_expression_variation_score, 2),
                "head_face_stability_score": round(stability_score, 2),
                "expression_variety_score": round(expression_variety_score, 2),
                "overall_score": round(overall, 2),
                # Backward-compatible aliases.
                "expressiveness_score": round(visible_expression_variation_score, 2),
                "posture_stability_score": round(stability_score, 2),
                "emotion_balance_score": round(expression_variety_score, 2),
            },
            "timeline_highlights": compute_timeline_highlights(self.records),
            "method_notes": [
                "Looking-forward is estimated from face position, nose/eye symmetry, and face roll; it is not real eye tracking.",
                "Head/face stability is based on face bounding-box and Face Mesh nose movement; it is not full-body posture analysis.",
                "Visible expression variation combines expression probability diversity with mouth/eye landmark variation.",
                "Scores are heuristic visual communication metrics for presentation practice.",
            ],
        }

    def save_frame_metrics_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(FrameMetric.__dataclass_fields__.keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(record.__dict__)

    def save_summary_json(self, path: Path, summary: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def save_charts(self, charts_dir: Path) -> dict[str, str]:
        charts_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "expression_timeline": charts_dir / "phase3_expression_timeline.png",
            "expression_distribution": charts_dir / "phase3_expression_distribution.png",
            "looking_forward": charts_dir / "phase3_looking_forward_over_time.png",
            "head_face_movement": charts_dir / "phase3_head_face_movement_over_time.png",
            "mouth_openness": charts_dir / "phase3_mouth_openness_over_time.png",
        }
        save_expression_timeline(self.records, paths["expression_timeline"])
        save_expression_distribution(self.records, paths["expression_distribution"])
        save_looking_forward_chart(self.records, paths["looking_forward"])
        save_movement_chart(self.records, paths["head_face_movement"])
        save_mouth_openness_chart(self.records, paths["mouth_openness"])
        return {k: str(v) for k, v in paths.items()}


def estimate_looking_forward_with_bbox(cx: float, cy: float, area_ratio: float) -> bool:
    """Fallback frontal-presence proxy when Face Mesh is unavailable."""
    return bool(0.30 <= cx <= 0.70 and 0.20 <= cy <= 0.82 and area_ratio >= 0.05)


def estimate_looking_forward_with_landmarks(
    bbox_center: tuple[float, float],
    area_ratio: float,
    yaw_proxy: float,
    roll_deg: float,
) -> bool:
    """Landmark-based looking-forward approximation.

    This is not eye tracking. It checks whether the face is centered, reasonably
    sized, not strongly rolled, and has the nose close to the midpoint between
    the eyes, which is a simple proxy for frontal face orientation.
    """
    cx, cy = bbox_center
    centered = 0.28 <= cx <= 0.72 and 0.18 <= cy <= 0.84
    reasonable_size = area_ratio >= 0.04
    roughly_frontal = abs(yaw_proxy) <= 0.22 and abs(roll_deg) <= 14.0
    return bool(centered and reasonable_size and roughly_frontal)


def classify_movement(movement: float) -> str:
    if movement <= 0.008:
        return "stable"
    if movement <= 0.030:
        return "moderate"
    return "high"


def compute_head_face_stability_score(movements: list[float]) -> float:
    if not movements:
        return 0.0
    mean_movement = float(np.mean(movements))
    score = 100.0 - mean_movement * 1800.0
    return float(np.clip(score, 0.0, 100.0))


def compute_expression_variety_score(percentages: dict[str, float]) -> float:
    """Score whether one displayed model cue dominates the recording.

    The distribution includes ``uncertain``. This is intentional: if the model is
    unsure for many frames, the score should reflect limited reliable expression
    variety instead of hiding those frames.
    """
    if not percentages:
        return 0.0
    max_pct = max(percentages.values())
    uncertain_pct = percentages.get("uncertain", 0.0)
    score = 100.0 - max_pct
    # A large uncertain share means the classifier is not providing reliable
    # expression categories, so reduce the score mildly.
    score -= 0.25 * uncertain_pct
    return float(np.clip(score, 0.0, 100.0))


def compute_visible_expression_variation_score(
    expression_percentages: dict[str, float],
    counts: Counter,
    confidences: list[float],
    mouth_values: list[float],
    eye_values: list[float],
) -> float:
    """Heuristic variation score from expression model + Face Mesh motion.

    It is intentionally named as visible expression variation, not true emotion
    or personality expressiveness.
    """
    if not counts:
        return 0.0

    neutral_pct = expression_percentages.get("neutral", 0.0)
    uncertain_pct = expression_percentages.get("uncertain", 0.0)
    unique_ratio = min(1.0, len(counts) / 5.0)
    # Do not treat uncertain frames as expressiveness.
    reliable_component = max(0.0, 100.0 - neutral_pct - uncertain_pct)
    confidence_variation = min(100.0, (float(np.std(confidences)) if confidences else 0.0) * 250.0)

    mouth_variation = min(100.0, (float(np.std(mouth_values)) if mouth_values else 0.0) * 1400.0)
    eye_variation = min(100.0, (float(np.std(eye_values)) if eye_values else 0.0) * 700.0)
    landmark_variation = 0.70 * mouth_variation + 0.30 * eye_variation

    # Keep expression classifier as the main signal, but ground the metric with
    # actual facial landmark motion when Face Mesh is available.
    score = (
        0.35 * reliable_component
        + 0.25 * (unique_ratio * 100.0)
        + 0.15 * confidence_variation
        + 0.25 * landmark_variation
    )
    return float(np.clip(score, 0.0, 100.0))


# Backward-compatible function names for older imports/tests.
def compute_stability_score(movements: list[float]) -> float:
    return compute_head_face_stability_score(movements)


def compute_emotion_balance_score(percentages: dict[str, float]) -> float:
    return compute_expression_variety_score(percentages)


def compute_expressiveness_score(percentages: dict[str, float], counts: Counter, confidences: list[float]) -> float:
    return compute_visible_expression_variation_score(percentages, counts, confidences, [], [])


def compute_timeline_highlights(records: list[FrameMetric]) -> dict:
    if not records:
        return {}
    high_movement = max(records, key=lambda r: r.movement_magnitude)
    highest_conf = max(records, key=lambda r: r.expression_confidence)
    low_conf = min((r for r in records if r.expression_confidence > 0), key=lambda r: r.expression_confidence, default=None)
    return {
        "strongest_expression_segment_sec": round(highest_conf.timestamp_sec, 2),
        "strongest_expression": highest_conf.expression,
        "strongest_expression_confidence": round(highest_conf.expression_confidence, 4),
        "lowest_confidence_segment_sec": round(low_conf.timestamp_sec, 2) if low_conf else None,
        "highest_head_face_movement_segment_sec": round(high_movement.timestamp_sec, 2),
        "max_movement_magnitude": round(high_movement.movement_magnitude, 6),
        # Backward-compatible alias.
        "unstable_movement_segment_sec": round(high_movement.timestamp_sec, 2),
    }


def empty_summary() -> dict:
    return {
        "video_info": {"duration_sec": 0, "fps": 0, "analyzed_frames": 0, "face_detection_rate": 0},
        "expression_summary": {"dominant_expression": "not_available", "average_confidence": 0},
        "emotion_summary": {"dominant_expression": "not_available", "average_confidence": 0},
        "expression_percentages": {},
        "emotion_percentages": {},
        "head_face_geometry": {},
        "movement": {"mean_movement": 0, "max_movement": 0},
        "scores": {
            "looking_forward_score": 0,
            "visible_expression_variation_score": 0,
            "head_face_stability_score": 0,
            "expression_variety_score": 0,
            "overall_score": 0,
        },
        "timeline_highlights": {},
        "method_notes": [],
    }


def save_expression_timeline(records: list[FrameMetric], path: Path) -> None:
    valid = [r for r in records if r.expression not in {"no_face", "not_available"}]
    if not valid:
        return save_empty_chart(path, "Expression Timeline")
    labels = sorted({r.expression for r in valid})
    label_to_idx = {label: i for i, label in enumerate(labels)}
    x = [r.timestamp_sec for r in valid]
    y = [label_to_idx[r.expression] for r in valid]
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker="o", linewidth=1, markersize=2)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Visible expression cue")
    plt.title("Expression Timeline")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_expression_distribution(records: list[FrameMetric], path: Path) -> None:
    labels = [r.expression for r in records if r.expression not in {"no_face", "not_available"}]
    if not labels:
        return save_empty_chart(path, "Expression Distribution")
    counts = Counter(labels)
    keys = list(counts.keys())
    values = [counts[k] for k in keys]
    plt.figure(figsize=(8, 4))
    plt.bar(keys, values)
    plt.xlabel("Expression")
    plt.ylabel("Frame count")
    plt.title("Expression Distribution")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_looking_forward_chart(records: list[FrameMetric], path: Path) -> None:
    valid = [r for r in records if r.looking_forward is not None]
    if not valid:
        return save_empty_chart(path, "Looking-Forward Approximation Over Time")
    x = [r.timestamp_sec for r in valid]
    y = [1 if r.looking_forward else 0 for r in valid]
    plt.figure(figsize=(10, 3))
    plt.plot(x, y, linewidth=1)
    plt.yticks([0, 1], ["No", "Yes"])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Looking-forward")
    plt.title("Looking-Forward Approximation Over Time")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_movement_chart(records: list[FrameMetric], path: Path) -> None:
    valid = [r for r in records if r.face_detected]
    if not valid:
        return save_empty_chart(path, "Head/Face Movement Over Time")
    x = [r.timestamp_sec for r in valid]
    y = [r.movement_magnitude for r in valid]
    plt.figure(figsize=(10, 3))
    plt.plot(x, y, linewidth=1)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized movement")
    plt.title("Head/Face Movement Over Time")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_mouth_openness_chart(records: list[FrameMetric], path: Path) -> None:
    valid = [r for r in records if r.face_mesh_detected]
    if not valid:
        return save_empty_chart(path, "Mouth Openness Over Time")
    x = [r.timestamp_sec for r in valid]
    y = [r.mouth_open_ratio for r in valid]
    plt.figure(figsize=(10, 3))
    plt.plot(x, y, linewidth=1)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mouth openness ratio")
    plt.title("Mouth Openness Over Time")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# Backward-compatible chart function names.
def save_emotion_timeline(records: list[FrameMetric], path: Path) -> None:
    save_expression_timeline(records, path)


def save_emotion_distribution(records: list[FrameMetric], path: Path) -> None:
    save_expression_distribution(records, path)


def save_empty_chart(path: Path, title: str) -> None:
    plt.figure(figsize=(6, 3))
    plt.title(title)
    plt.text(0.5, 0.5, "No data available", ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
