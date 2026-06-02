"""PresentSense Streamlit application.

This app is intentionally a thin, product-style interface around the existing
`analyze_video.py` CLI. Keeping the analysis in the CLI prevents the Streamlit
app from breaking the already-working webcam/video pipeline.
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
APP_DIR = OUTPUTS / "app"
APP_UPLOADS = APP_DIR / "uploads"
APP_VIDEOS = APP_DIR / "videos"
APP_REPORTS = APP_DIR / "reports"
APP_CHARTS = APP_DIR / "charts"
DEFAULT_MODEL = ROOT / "models" / "best_exp03_resnet18_finetune.pth"

CHART_FILES = {
    "Expression Distribution": "phase3_expression_distribution.png",
    "Expression Timeline": "phase3_expression_timeline.png",
    "Looking-Forward Approximation": "phase3_looking_forward_over_time.png",
    "Head/Face Movement": "phase3_head_face_movement_over_time.png",
    "Mouth Openness": "phase3_mouth_openness_over_time.png",
    "Eye Openness": "phase3_eye_openness_over_time.png",
}


@dataclass
class AppSettings:
    model_path: Path
    uncertainty_threshold: float
    frame_step: int
    use_face_mesh: bool
    save_annotated_video: bool
    debug_overlay: bool


def ensure_app_dirs() -> None:
    for path in [APP_DIR, APP_UPLOADS, APP_VIDEOS, APP_REPORTS, APP_CHARTS]:
        path.mkdir(parents=True, exist_ok=True)


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def copy_if_exists(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def archive_latest_outputs(video_output: Path | None = None) -> dict[str, Path]:
    """Copy latest CLI outputs into outputs/app/ for Streamlit display/download."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archived: dict[str, Path] = {}

    report_map = {
        "summary": (OUTPUTS / "reports" / "summary_report.json", APP_REPORTS / f"summary_report_{timestamp}.json"),
        "report": (OUTPUTS / "reports" / "presentation_report.md", APP_REPORTS / f"presentation_report_{timestamp}.md"),
        "frame_metrics": (OUTPUTS / "reports" / "frame_metrics.csv", APP_REPORTS / f"frame_metrics_{timestamp}.csv"),
    }
    for key, (src, dst) in report_map.items():
        copied = copy_if_exists(src, dst)
        if copied:
            archived[key] = copied

    for title, filename in CHART_FILES.items():
        src = OUTPUTS / "charts" / filename
        dst = APP_CHARTS / f"{Path(filename).stem}_{timestamp}.png"
        copied = copy_if_exists(src, dst)
        if copied:
            archived[Path(filename).stem] = copied

    if video_output and video_output.exists():
        dst = APP_VIDEOS / f"annotated_video_{timestamp}{video_output.suffix}"
        copied = copy_if_exists(video_output, dst)
        if copied:
            archived["video"] = copied

    # Also maintain stable "latest" copies for the Latest Results tab.
    latest_map = {
        OUTPUTS / "reports" / "summary_report.json": APP_REPORTS / "latest_summary_report.json",
        OUTPUTS / "reports" / "presentation_report.md": APP_REPORTS / "latest_presentation_report.md",
        OUTPUTS / "reports" / "frame_metrics.csv": APP_REPORTS / "latest_frame_metrics.csv",
    }
    for src, dst in latest_map.items():
        copy_if_exists(src, dst)
    for _, filename in CHART_FILES.items():
        copy_if_exists(OUTPUTS / "charts" / filename, APP_CHARTS / filename)
    if video_output and video_output.exists():
        copy_if_exists(video_output, APP_VIDEOS / "latest_annotated_video.mp4")

    return archived


def build_analysis_command(source: Path | str, settings: AppSettings, output_video: Path | None) -> list[str]:
    cmd = [
        sys.executable,
        "analyze_video.py",
        "--source",
        str(source),
        "--model",
        str(settings.model_path),
        "--frame-step",
        str(settings.frame_step),
        "--uncertainty-threshold",
        f"{settings.uncertainty_threshold:.2f}",
    ]

    if output_video is not None:
        cmd.extend(["--output", str(output_video)])

    # These flags are supported by the Phase 3.5 patch. If an older CLI does not
    # support them, run_command() will show the CLI error clearly in Streamlit.
    if not settings.use_face_mesh:
        cmd.append("--no-face-mesh")
    if settings.debug_overlay:
        cmd.append("--debug-overlay")

    return cmd


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError as exc:
        return 1, "", f"Command not found: {exc}"
    except Exception as exc:
        return 1, "", f"Unexpected error: {exc}"


def save_uploaded_file(uploaded_file: Any) -> Path:
    ensure_app_dirs()
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    safe_stem = Path(uploaded_file.name).stem.replace(" ", "_")[:60]
    target = APP_UPLOADS / f"{safe_stem}_{int(time.time())}{suffix}"
    with target.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def score_value(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get("scores", {}).get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def fmt_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}/100"


def show_summary_dashboard(summary: dict[str, Any]) -> None:
    video_info = summary.get("video_info", {})
    expression_summary = summary.get("expression_summary", summary.get("emotion_summary", {}))

    st.subheader("Results Dashboard")
    st.caption("Scores are heuristic visual communication cues, not absolute judgments.")

    cols = st.columns(4)
    with cols[0]:
        metric_card("Overall Score", fmt_score(score_value(summary, "overall_score")))
    with cols[1]:
        metric_card("Looking Forward", fmt_score(score_value(summary, "looking_forward_score")), "Approximation based on face position and Face Mesh geometry.")
    with cols[2]:
        metric_card("Head/Face Stability", fmt_score(score_value(summary, "head_face_stability_score")), "Based on face box and landmark motion, not full-body posture.")
    with cols[3]:
        metric_card("Expression Variation", fmt_score(score_value(summary, "visible_expression_variation_score")), "Combines model confidence variation and facial landmark movement.")

    cols = st.columns(4)
    with cols[0]:
        metric_card("Expression Variety", fmt_score(score_value(summary, "expression_variety_score")))
    with cols[1]:
        metric_card("Face Visibility", f"{video_info.get('face_detection_rate', 0):.1f}%")
    with cols[2]:
        metric_card("Face Mesh Detection", f"{video_info.get('face_mesh_detection_rate', 0):.1f}%")
    with cols[3]:
        avg_conf = expression_summary.get("average_confidence")
        metric_card("Avg. Model Confidence", "N/A" if avg_conf is None else f"{float(avg_conf):.2f}")

    with st.expander("Video details"):
        st.markdown(
            f"""
- **Duration:** {video_info.get("duration_sec", "N/A")} seconds
- **FPS:** {video_info.get("fps", "N/A")}
- **Analyzed frames:** {video_info.get("analyzed_frames", "N/A")}
- **Face detected frames:** {video_info.get("face_detected_frames", "N/A")}
- **Face Mesh detected frames:** {video_info.get("face_mesh_detected_frames", "N/A")}
- **Dominant model-predicted cue:** {expression_summary.get("dominant_expression", "N/A")}
"""
        )


def split_recommendations(report_text: str | None, summary: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    """Generate friendly buckets from report recommendations and scores."""
    went_well: list[str] = []
    improve: list[str] = []

    if summary:
        scores = summary.get("scores", {})
        video_info = summary.get("video_info", {})
        expr = summary.get("expression_percentages", summary.get("emotion_percentages", {}))

        if scores.get("looking_forward_score", 0) >= 75:
            went_well.append("You stayed visually oriented toward the camera for most of the recording.")
        else:
            improve.append("Try to face the camera/audience more consistently during key points.")

        if scores.get("head_face_stability_score", 0) >= 75:
            went_well.append("Your head/face movement was stable, which helps the audience focus.")
        else:
            improve.append("Reduce unnecessary head movement and pause briefly between transitions.")

        if scores.get("visible_expression_variation_score", 0) >= 65:
            went_well.append("Your visible facial cues changed enough to avoid looking static.")
        else:
            improve.append("Try adding more natural facial emphasis when introducing or concluding important ideas.")

        uncertain_pct = float(expr.get("uncertain", 0) or 0)
        if uncertain_pct >= 30:
            improve.append("Many frames were uncertain. Improve front lighting, camera angle, and face visibility before interpreting expression cues strongly.")

        if video_info.get("face_detection_rate", 0) < 80:
            improve.append("Your face was not visible for part of the recording. Keep the camera at eye level and stay within frame.")

    # Add report recommendations as fallback, avoiding duplicates.
    if report_text:
        in_recs = False
        for line in report_text.splitlines():
            if line.strip().startswith("## 7. Recommendations") or line.strip().startswith("## 6. Recommendations"):
                in_recs = True
                continue
            if in_recs and line.startswith("## "):
                break
            if in_recs and line.strip().startswith("-"):
                rec = line.strip().lstrip("- ").strip()
                target = went_well if any(word in rec.lower() for word in ["good", "strong", "positive", "stable"]) else improve
                if rec not in went_well and rec not in improve:
                    target.append(rec)

    if not went_well:
        went_well.append("The analysis completed successfully and your face was detected in the recording.")
    if not improve:
        improve.append("Keep practicing with consistent lighting, camera at eye level, and clear face visibility.")

    return went_well[:5], improve[:5]


def show_feedback(report_text: str | None, summary: dict[str, Any] | None) -> None:
    st.subheader("Feedback")
    went_well, improve = split_recommendations(report_text, summary)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What went well")
        for item in went_well:
            st.success(item)
    with col2:
        st.markdown("### What to improve")
        for item in improve:
            st.info(item)


def show_charts(chart_dir: Path = APP_CHARTS) -> None:
    st.subheader("Detailed Charts")
    captions = {
        "Expression Distribution": "How often each model-predicted visible cue appeared. `uncertain` means confidence was below the threshold.",
        "Expression Timeline": "Visible cue estimates over time. Use this as a trend, not as a psychological interpretation.",
        "Looking-Forward Approximation": "Whether the face appeared oriented toward the camera based on face position and Face Mesh geometry.",
        "Head/Face Movement": "Normalized movement of the face box and facial landmarks over time.",
        "Mouth Openness": "Mouth openness estimated from Face Mesh landmarks. Useful for visual expressiveness, not speech understanding.",
        "Eye Openness": "Eye openness estimated from Face Mesh landmarks, if generated by the pipeline.",
    }

    for title, filename in CHART_FILES.items():
        path = chart_dir / filename
        if not path.exists():
            # fallback to root outputs/charts
            path = OUTPUTS / "charts" / filename
        if path.exists():
            st.markdown(f"#### {title}")
            st.image(str(path), caption=captions.get(title, ""), use_container_width=True)


def download_button_for_file(label: str, path: Path, mime: str, key: str) -> None:
    """Render a uniquely-keyed download button only when the file exists."""
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            key=key,
        )


def build_analysis_zip(video_path: Path | None = None) -> bytes:
    """Create an in-memory ZIP with the latest analysis outputs."""
    buffer = io.BytesIO()
    candidate_video = video_path or APP_VIDEOS / "latest_annotated_video.mp4"

    files: list[tuple[Path, str]] = [
        (APP_REPORTS / "latest_presentation_report.md", "reports/presentation_report.md"),
        (APP_REPORTS / "latest_summary_report.json", "reports/summary_report.json"),
        (APP_REPORTS / "latest_frame_metrics.csv", "reports/frame_metrics.csv"),
    ]

    for _, filename in CHART_FILES.items():
        chart_path = APP_CHARTS / filename
        if chart_path.exists():
            files.append((chart_path, f"charts/{filename}"))

    if candidate_video.exists():
        files.append((candidate_video, f"videos/{candidate_video.name}"))

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in files:
            if path.exists():
                zf.write(path, arcname)

    buffer.seek(0)
    return buffer.getvalue()


def clear_app_outputs() -> None:
    """Clear app-generated files and stale root-level generated outputs.

    This keeps .gitkeep files and curated Phase 3 folders, but removes temporary
    files created by the Streamlit workflow and by the latest CLI run.
    """
    ensure_app_dirs()

    # Clear app workspace.
    for folder in [APP_UPLOADS, APP_VIDEOS, APP_REPORTS, APP_CHARTS]:
        for item in folder.iterdir():
            if item.name == ".gitkeep":
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)

    # Clear root-level temporary outputs generated by analyze_video.py.
    # Do not touch curated folders such as outputs/charts/phase3 or outputs/reports/phase3.
    for path in [
        OUTPUTS / "reports" / "summary_report.json",
        OUTPUTS / "reports" / "presentation_report.md",
        OUTPUTS / "reports" / "frame_metrics.csv",
    ]:
        path.unlink(missing_ok=True)

    charts_dir = OUTPUTS / "charts"
    if charts_dir.exists():
        for path in charts_dir.glob("phase3_*.png"):
            if path.is_file():
                path.unlink(missing_ok=True)

    for key in ["latest_video", "latest_summary", "latest_report"]:
        st.session_state.pop(key, None)

    st.session_state["upload_widget_key"] = st.session_state.get("upload_widget_key", 0) + 1


def show_downloads(video_path: Path | None = None, key_prefix: str = "main") -> None:
    st.subheader("Download analysis")
    st.caption("Download everything as one ZIP, or open individual files below.")

    zip_bytes = build_analysis_zip(video_path)
    if zip_bytes:
        st.download_button(
            label="Download full analysis ZIP",
            data=zip_bytes,
            file_name=f"presentsense_analysis_{time.strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
            key=f"{key_prefix}_download_full_analysis_zip",
        )
    else:
        st.info("No downloadable analysis files found yet.")

    with st.expander("Download individual files"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            download_button_for_file(
                "Markdown report",
                APP_REPORTS / "latest_presentation_report.md",
                "text/markdown",
                key=f"{key_prefix}_download_markdown_report",
            )
        with col2:
            download_button_for_file(
                "Summary JSON",
                APP_REPORTS / "latest_summary_report.json",
                "application/json",
                key=f"{key_prefix}_download_summary_json",
            )
        with col3:
            download_button_for_file(
                "Frame metrics CSV",
                APP_REPORTS / "latest_frame_metrics.csv",
                "text/csv",
                key=f"{key_prefix}_download_frame_metrics_csv",
            )
        with col4:
            candidate = video_path or APP_VIDEOS / "latest_annotated_video.mp4"
            if candidate.exists():
                st.download_button(
                    label="Annotated video",
                    data=candidate.read_bytes(),
                    file_name=candidate.name,
                    mime="video/mp4",
                    key=f"{key_prefix}_download_annotated_video",
                )

    st.divider()
    st.markdown("### Analyze another video")
    st.write("Clear the current app-generated files and reset the uploader before analyzing a new video.")
    if st.button(
        "Clear current analysis and upload another video",
        type="secondary",
        use_container_width=True,
        key=f"{key_prefix}_clear_current_analysis",
    ):
        clear_app_outputs()
        st.success("Current app outputs were cleared. You can upload another video now.")
        st.rerun()

def show_annotated_video(video_path: Path | None = None) -> None:
    st.subheader("Annotated Video")
    candidate = video_path or APP_VIDEOS / "latest_annotated_video.mp4"
    if candidate.exists():
        st.video(str(candidate))
    else:
        st.info("No annotated video found yet. Enable 'Save annotated video' and run an analysis.")


def show_limitations() -> None:
    with st.expander("Limitations and ethics"):
        st.markdown(
            """
- PresentSense analyzes visual cues only.
- It is not an emotion, psychological, clinical, personality, confidence, or mental-health diagnosis tool.
- Model-predicted expression cues can be wrong, especially with poor lighting, camera angle, occlusion, or natural expressions.
- `uncertain` means the model confidence was below the selected threshold; it is not a negative judgment.
- Looking-forward is an approximation based on face position and Face Mesh geometry. It is not real eye tracking.
- Head/face stability is based on face and landmark movement. It is not full-body posture analysis.
- Audio, speech quality, slide content, and presentation structure are not analyzed in this version.
- Process videos locally when possible to protect privacy.
            """.strip()
        )


def sidebar_settings() -> AppSettings:
    st.sidebar.title("PresentSense")
    st.sidebar.caption("Presentation visual feedback")

    model_input = st.sidebar.text_input("Model path", value=rel_or_abs(DEFAULT_MODEL))
    uncertainty_threshold = st.sidebar.slider("Uncertainty threshold", 0.40, 0.75, 0.60, 0.01)
    frame_step = st.sidebar.slider("Frame step", 1, 5, 1, 1)
    use_face_mesh = st.sidebar.toggle("Use Face Mesh", value=True)
    save_annotated_video = st.sidebar.toggle("Save annotated video", value=True)
    debug_overlay = st.sidebar.toggle("Debug overlay", value=False)

    st.sidebar.divider()
    st.sidebar.info("Privacy note: videos are processed locally in this project environment.")

    return AppSettings(
        model_path=(ROOT / model_input).resolve() if not Path(model_input).is_absolute() else Path(model_input),
        uncertainty_threshold=uncertainty_threshold,
        frame_step=frame_step,
        use_face_mesh=use_face_mesh,
        save_annotated_video=save_annotated_video,
        debug_overlay=debug_overlay,
    )


def render_home() -> None:
    st.title("PresentSense: Presentation Visual Feedback")
    st.write(
        "Upload your practice presentation and get visual feedback about face visibility, "
        "looking-forward behavior, head/face stability, and visible expression cues."
    )
    st.warning(
        "PresentSense is a visual-only presentation practice tool. It is not a medical, "
        "psychological, emotion, confidence, or personality diagnosis system."
    )

    with st.expander("How it works"):
        st.markdown(
            """
1. The app saves your uploaded video locally.
2. It calls the existing `analyze_video.py` pipeline.
3. The pipeline detects the face, estimates visible expression cues, tracks Face Mesh geometry, and creates reports.
4. The app displays a friendly dashboard, charts, recommendations, and downloads.
            """.strip()
        )


def analyze_uploaded_video(uploaded_file: Any, settings: AppSettings) -> None:
    if uploaded_file is None:
        st.info("Upload a video first.")
        return
    if not settings.model_path.exists():
        st.error(f"Model not found: {settings.model_path}")
        return

    input_path = save_uploaded_file(uploaded_file)
    output_path = APP_VIDEOS / f"annotated_{input_path.stem}.mp4" if settings.save_annotated_video else None
    cmd = build_analysis_command(input_path, settings, output_path)

    with st.spinner("Analyzing your presentation video..."):
        code, stdout, stderr = run_command(cmd)

    with st.expander("Analysis command and logs"):
        st.code(" ".join(cmd), language="bash")
        if stdout:
            st.text_area("stdout", stdout, height=160)
        if stderr:
            st.text_area("stderr", stderr, height=160)

    if code != 0:
        st.error("Analysis failed. Check the logs above.")
        return

    archived = archive_latest_outputs(output_path)
    st.success("Analysis complete.")
    st.session_state["latest_video"] = str(archived.get("video", output_path or ""))
    st.session_state["latest_summary"] = str(APP_REPORTS / "latest_summary_report.json")
    st.session_state["latest_report"] = str(APP_REPORTS / "latest_presentation_report.md")


def run_webcam_and_load_results(settings: AppSettings) -> None:
    """Run the webcam pipeline and load results after the OpenCV window is closed.

    This intentionally blocks the Streamlit script while the OpenCV webcam window
    is open. When the user presses q or ESC in that window, analyze_video.py
    finishes, writes the standard Phase 3 reports/charts, and this function copies
    them into outputs/app/ so the Streamlit dashboard can display them.
    """
    if not settings.model_path.exists():
        st.error(f"Model not found: {settings.model_path}")
        return

    output_path = APP_VIDEOS / f"webcam_demo_{int(time.time())}.mp4" if settings.save_annotated_video else None
    cmd = build_analysis_command("webcam", settings, output_path)

    with st.spinner("Webcam analysis is running. Stop the OpenCV window with q or ESC to load the results here..."):
        code, stdout, stderr = run_command(cmd)

    with st.expander("Webcam command and logs"):
        st.code(" ".join(cmd), language="bash")
        if stdout:
            st.text_area("stdout", stdout, height=160, key="webcam_stdout_logs")
        if stderr:
            st.text_area("stderr", stderr, height=160, key="webcam_stderr_logs")

    if code != 0:
        st.error("Webcam analysis failed. Check the logs above.")
        return

    archived = archive_latest_outputs(output_path)
    st.session_state["latest_video"] = str(archived.get("video", output_path or ""))
    st.session_state["latest_summary"] = str(APP_REPORTS / "latest_summary_report.json")
    st.session_state["latest_report"] = str(APP_REPORTS / "latest_presentation_report.md")
    st.success("Webcam analysis complete. The dashboard below has been updated with the latest webcam results.")


def launch_webcam_background(settings: AppSettings) -> None:
    """Optional non-blocking webcam launch for users who only want the OpenCV demo."""
    if not settings.model_path.exists():
        st.error(f"Model not found: {settings.model_path}")
        return

    output_path = APP_VIDEOS / f"webcam_demo_{int(time.time())}.mp4" if settings.save_annotated_video else None
    cmd = build_analysis_command("webcam", settings, output_path)

    try:
        subprocess.Popen(cmd, cwd=ROOT)
        st.session_state["pending_webcam_video"] = str(output_path or "")
        st.success("Webcam demo launched in a separate OpenCV window. Press q or ESC in that window to stop.")
        st.info("Because this mode runs in the background, click 'Load latest webcam results' after closing the OpenCV window.")
        st.code(" ".join(cmd), language="bash")
    except Exception as exc:
        st.error(f"Could not launch webcam demo: {exc}")


def load_latest_webcam_results() -> None:
    """Archive outputs generated by a background webcam run."""
    pending = st.session_state.get("pending_webcam_video")
    pending_path = Path(pending) if pending else APP_VIDEOS / "latest_annotated_video.mp4"
    archived = archive_latest_outputs(pending_path if pending_path.exists() else None)
    st.session_state["latest_video"] = str(archived.get("video", APP_VIDEOS / "latest_annotated_video.mp4"))
    st.session_state["latest_summary"] = str(APP_REPORTS / "latest_summary_report.json")
    st.session_state["latest_report"] = str(APP_REPORTS / "latest_presentation_report.md")
    if (APP_REPORTS / "latest_summary_report.json").exists() or (APP_REPORTS / "latest_presentation_report.md").exists():
        st.success("Latest webcam outputs loaded into the dashboard.")
    else:
        st.warning("No webcam report files were found yet. Make sure you stopped the OpenCV window with q or ESC.")


def render_results(video_path: Path | None = None, key_prefix: str = "main") -> None:
    summary = read_json(APP_REPORTS / "latest_summary_report.json") or read_json(OUTPUTS / "reports" / "summary_report.json")
    report_text = read_text(APP_REPORTS / "latest_presentation_report.md") or read_text(OUTPUTS / "reports" / "presentation_report.md")

    if not summary and not report_text:
        st.info("No results found yet. Analyze an uploaded video first or load outputs generated by the CLI.")
        return

    if summary:
        show_summary_dashboard(summary)
    show_feedback(report_text, summary)
    show_annotated_video(video_path)
    show_charts(APP_CHARTS)

    st.subheader("Presentation Report")
    if report_text:
        st.markdown(report_text)
    else:
        st.info("No Markdown report found.")

    show_downloads(video_path, key_prefix=key_prefix)


def main() -> None:
    st.set_page_config(
        page_title="PresentSense",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_app_dirs()
    st.session_state.setdefault("upload_widget_key", 0)
    settings = sidebar_settings()
    render_home()

    tab_upload, tab_webcam, tab_latest = st.tabs(["Upload Video", "Webcam Demo", "Latest Results"])

    with tab_upload:
        st.header("Upload your practice presentation")
        st.write("Supported formats: MP4, MOV, AVI, WEBM. Short videos are recommended for fast analysis.")
        uploaded_file = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "webm"], key=f"video_upload_{st.session_state['upload_widget_key']}")
        if uploaded_file:
            st.caption(f"Selected: {uploaded_file.name}")
        if st.button("Analyze Uploaded Video", type="primary", key="analyze_uploaded_video_button"):
            analyze_uploaded_video(uploaded_file, settings)
        render_results(Path(st.session_state["latest_video"]) if st.session_state.get("latest_video") else None, key_prefix="upload_results")

    with tab_webcam:
        st.header("Webcam demo")
        st.write("Use your real webcam through the existing OpenCV pipeline.")
        st.info("Press q or ESC inside the OpenCV window to stop the webcam demo and load.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Launch Webcam Demo", type="primary", key="record_webcam_and_load_results_button"):
                run_webcam_and_load_results(settings)

        render_results(
            Path(st.session_state["latest_video"]) if st.session_state.get("latest_video") else None,
            key_prefix="webcam_results",
        )

        with st.expander("Why does webcam open in a separate window?"):
            st.write(
                "The real-time webcam pipeline is implemented with OpenCV. "
                "The browser app controls the workflow, but OpenCV opens the local camera window. "
                "When the OpenCV window closes, the app loads the generated reports and charts."
            )

    with tab_latest:
        st.header("Latest generated results")
        if st.button("Refresh latest outputs", key="refresh_latest_outputs_button"):
            archive_latest_outputs(APP_VIDEOS / "latest_annotated_video.mp4")
        render_results(key_prefix="latest_results")

    show_limitations()


if __name__ == "__main__":
    main()
