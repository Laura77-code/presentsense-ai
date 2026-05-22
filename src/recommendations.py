"""Recommendation rules for PresentSense visual feedback reports."""

from __future__ import annotations


def generate_recommendations(summary: dict) -> list[str]:
    """Generate 3-6 supportive presentation-practice recommendations.

    These recommendations are heuristic and are not psychological judgments.
    """
    recommendations: list[str] = []

    scores = summary.get("scores", {})
    expression_percentages = summary.get("expression_percentages", summary.get("emotion_percentages", {}))
    movement = summary.get("movement", {})
    geometry = summary.get("head_face_geometry", {})

    looking = float(scores.get("looking_forward_score", 0.0))
    variation = float(scores.get("visible_expression_variation_score", scores.get("expressiveness_score", 0.0)))
    stability = float(scores.get("head_face_stability_score", scores.get("posture_stability_score", 0.0)))
    variety = float(scores.get("expression_variety_score", scores.get("emotion_balance_score", 0.0)))
    overall = float(scores.get("overall_score", 0.0))
    neutral_pct = float(expression_percentages.get("neutral", 0.0))
    uncertain_pct = float(expression_percentages.get("uncertain", 0.0))
    happy_pct = float(expression_percentages.get("happy", 0.0))
    mean_movement = float(movement.get("mean_movement", 0.0))
    mouth_variation = float(geometry.get("mouth_open_variation", 0.0))

    if looking < 50:
        recommendations.append("Try to keep your face oriented toward the camera/audience more consistently during key points.")
    elif looking >= 75:
        recommendations.append("Good looking-forward consistency. Your face stayed visually oriented toward the camera/audience for much of the recording.")

    if uncertain_pct > 35:
        recommendations.append("Many frames were classified as uncertain. Improve front lighting, camera angle, and face visibility before interpreting expression cues strongly.")

    if neutral_pct > 75 and uncertain_pct <= 35:
        recommendations.append("Your visible expression was mostly neutral. Add more facial emphasis when introducing important ideas or transitions.")

    if variation < 45:
        recommendations.append("Increase visible expression variation slightly by emphasizing key points with natural facial movement and clearer mouth articulation.")
    elif variation >= 70:
        recommendations.append("Good visible expression variation. Your facial cues changed enough to support communication without looking static.")

    if mouth_variation < 0.005 and variation < 60:
        recommendations.append("Mouth movement variation was limited. Practice articulating important words clearly while keeping the delivery natural.")

    if stability < 45 or mean_movement > 0.045:
        recommendations.append("Reduce unnecessary head movement and pause briefly between transitions to look more stable on camera.")
    elif stability >= 70:
        recommendations.append("Good head/face stability. Your face position was controlled enough for a clear camera presentation.")

    if variety < 35 and neutral_pct <= 75:
        recommendations.append("The expression distribution was dominated by one visible cue. Try varying emphasis naturally across opening, explanation, and conclusion.")

    if happy_pct > 10 and uncertain_pct < 50:
        recommendations.append("Positive/warm model-predicted cues appeared during the recording, which can help create a friendly presentation tone.")

    if overall < 50:
        recommendations.append("Practice a short 1-minute version while focusing on three basics: face the camera, reduce excessive movement, and emphasize key points visually.")
    elif overall >= 70:
        recommendations.append("Overall visual delivery looks strong. Continue practicing with consistent lighting and a camera at eye level.")

    unique: list[str] = []
    for item in recommendations:
        if item not in unique:
            unique.append(item)

    if not unique:
        unique.append("Continue practicing with good front lighting, a centered camera position, and natural facial emphasis.")
        unique.append("Use this report as supportive feedback, not as an absolute score of presentation ability.")

    return unique[:6]
