import streamlit as st
from PIL import Image
from collections import Counter
import io, tempfile, cv2, pandas as pd
from datetime import datetime

from shared_functions import (
    analyze_single_image,
    chat_with_model,
    load_ground_truth,
    calculate_metrics,
    display_confusion_matrix,
)
from config import PROMPTS, SYSTEM_PROMPT, MODEL_CONFIGS

st.set_page_config(page_title="Deepfake Detector", layout="wide", page_icon="🕵️‍♂️")

# session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "media" not in st.session_state:
    st.session_state.media = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = []

# build model selection mapping
model_key_to_display = {
    key: conf["display_name"] for key, conf in MODEL_CONFIGS.items()
}
display_to_model_key = {v: k for k, v in model_key_to_display.items()}

# main tabs
tab1, tab2 = st.tabs(["🔍 Detection", "📊 Evaluation"])

# ───────────────────────── Tab 1: Single image ─────────────────────────
with tab1:
    st.header("🕵️‍♂️ Deepfake Detection Chat")

    # model selector for detection
    detect_model_display = st.selectbox(
        "Select detection model",
        list(display_to_model_key.keys()),
        index=0,
    )
    detect_model_key = display_to_model_key[detect_model_display]

    left_col, right_col = st.columns([1, 2], gap="large")

    uploaded_file = right_col.file_uploader(
        "Upload image/video",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
        label_visibility="collapsed",
    )

    analysis_image = None
    new_upload = False

    if uploaded_file is not None:
        if (
            "last_uploaded" not in st.session_state
            or st.session_state.last_uploaded != uploaded_file.name
        ):
            st.session_state.last_uploaded = uploaded_file.name
            new_upload = True

            media_bytes = uploaded_file.read()
            if "image" in uploaded_file.type:
                analysis_image = Image.open(io.BytesIO(media_bytes))
                st.session_state.media = analysis_image
                st.session_state.messages.append(
                    {"role": "user", "content": f"📷 Uploaded image: {uploaded_file.name}"}
                )
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(media_bytes)
                    tmp_path = tmp.name
                cap = cv2.VideoCapture(tmp_path)
                success, frame = cap.read()
                cap.release()
                if success:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    analysis_image = Image.fromarray(frame_rgb)
                    st.session_state.media = analysis_image
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": f"🎞️ Uploaded video: {uploaded_file.name}",
                        }
                    )
                else:
                    st.error("Could not extract frames from video.")

    # media panel
    with left_col:
        st.markdown('<div id="media-panel">', unsafe_allow_html=True)
        if st.session_state.media is not None:
            st.image(st.session_state.media, use_container_width=True)
        else:
            st.info("Upload image/video to begin analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

    # chat panel
    with right_col:
        st.markdown('<div id="chat-panel">', unsafe_allow_html=True)
        st.markdown('<div id="chat-messages">', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            role_class = "user" if msg["role"] == "user" else "assistant"
            st.markdown(
                f"<div class='chat-message {role_class}'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        user_input = st.chat_input("Type a message...")

        st.markdown("</div>", unsafe_allow_html=True)

    if analysis_image is None:
        analysis_image = st.session_state.get("media")

    if new_upload and analysis_image:
        with st.spinner("🔍 Analyzing uploaded file..."):
            result = analyze_single_image(
                analysis_image, PROMPTS, SYSTEM_PROMPT, detect_model_key
            )
            assistant_msg = f"**Model:** {detect_model_display}\n\n**Analysis:**\n{result['analysis']}\n\n**Classification:** {result['classification']} (score={result['score']})"
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_msg}
            )

        st.rerun()

    elif user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("💬 Thinking..."):
            response = chat_with_model(
                st.session_state.messages, SYSTEM_PROMPT, detect_model_key
            )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# ───────────────────────── Tab 2: Batch Evaluation ─────────────────────────
with tab2:
    st.header("📊 Batch Evaluation")

    eval_images = st.file_uploader(
        "Upload images for evaluation",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    gt_file = st.file_uploader("Upload ground truth CSV", type=["csv"])

    num_runs = st.selectbox(
        "How many times would you like to process each image?",
        options=[1, 3, 5, 7, 9],
        index=2,
    )

    # choose which models to evaluate (default: all)
    model_multiselect = st.multiselect(
        "Select models to evaluate",
        options=list(display_to_model_key.keys()),
        default=list(display_to_model_key.keys()),
    )
    models_to_run = [display_to_model_key[d] for d in model_multiselect]

    if eval_images and gt_file and models_to_run and st.button("🚀 Run Evaluation"):
        try:
            gt_df = load_ground_truth(gt_file)

            per_model_results = {}  # model_key -> list[per image dicts]
            per_model_metrics = []  # list of metrics dicts with model info

            total_steps = len(eval_images) * num_runs * len(models_to_run)
            progress_bar = st.progress(0)
            step = 0

            for model_key in models_to_run:
                model_display = MODEL_CONFIGS[model_key]["display_name"]
                st.write(f"### Running model: {model_display}")
                per_image_results = []

                for i, img_file in enumerate(eval_images):
                    img = Image.open(img_file)
                    filename = img_file.name

                    if filename not in gt_df["filename"].values:
                        st.warning(f"No ground truth for {filename}, skipping")
                        continue

                    actual = gt_df.loc[
                        gt_df["filename"] == filename, "label"
                    ].values[0]

                    votes = []
                    first_analysis = None
                    first_score = None

                    for j in range(num_runs):
                        res = analyze_single_image(
                            img, PROMPTS, SYSTEM_PROMPT, model_key
                        )

                        pred_label = (
                            "AI Generated"
                            if "AI" in res["classification"]
                            else "Real"
                        )
                        votes.append(pred_label)

                        if first_analysis is None:
                            first_analysis = res.get("analysis", "")
                            first_score = res.get("score", None)

                        step += 1
                        progress_bar.progress(step / total_steps)

                    vote_counts = Counter(votes)
                    consensus_label, consensus_count = vote_counts.most_common(1)[0]

                    ai_votes = vote_counts.get("AI Generated", 0)
                    real_votes = vote_counts.get("Real", 0)

                    correct = consensus_label == actual

                    per_image_results.append(
                        {
                            "model_key": model_key,
                            "model_name": model_display,
                            "filename": filename,
                            "actual_label": actual,
                            "consensus_label": consensus_label,
                            "ai_votes": ai_votes,
                            "real_votes": real_votes,
                            "total_runs": num_runs,
                            "correct": correct,
                            "analysis_example": first_analysis,
                            "score_example": first_score,
                        }
                    )

                if not per_image_results:
                    st.warning(
                        f"No valid images with ground truth were processed for model {model_display}."
                    )
                    continue

                per_model_results[model_key] = per_image_results

                y_true = [r["actual_label"] for r in per_image_results]
                y_pred = [r["consensus_label"] for r in per_image_results]
                metrics = calculate_metrics(y_true, y_pred)
                metrics["model_key"] = model_key
                metrics["model_name"] = model_display
                per_model_metrics.append(metrics)

            if not per_model_metrics:
                st.error("No models produced valid results.")
                st.stop()

            # summary metrics table
            metrics_df = pd.DataFrame(per_model_metrics)
            st.subheader("📈 Metrics by Model (Consensus per Image)")
            st.dataframe(
                metrics_df[
                    [
                        "model_name",
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "tp",
                        "tn",
                        "fp",
                        "fn",
                    ]
                ]
            )

            # confusion matrices per model (expanders)
            for m in per_model_metrics:
                with st.expander(f"Confusion Matrix – {m['model_name']}"):
                    display_confusion_matrix(m)

            # per-image predictions across all models
            all_pred_rows = []
            for model_key, res_list in per_model_results.items():
                all_pred_rows.extend(res_list)
            preds_df = pd.DataFrame(all_pred_rows)

            st.subheader("📋 Per-Image Consensus Results (All Models)")
            df_display_ui = preds_df.copy()
            if "analysis_example" in df_display_ui.columns:
                df_display_ui["analysis_example"] = (
                    df_display_ui["analysis_example"].str.slice(0, 100) + "..."
                )
            st.dataframe(
                df_display_ui[
                    [
                        "model_name",
                        "filename",
                        "actual_label",
                        "consensus_label",
                        "ai_votes",
                        "real_votes",
                        "total_runs",
                        "correct",
                        "score_example",
                        "analysis_example",
                    ]
                ]
            )

            # Excel export: metrics + predictions in 2 sheets
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                metrics_df.to_excel(writer, sheet_name="metrics", index=False)
                preds_df.to_excel(writer, sheet_name="predictions", index=False)
            excel_buf.seek(0)

            st.download_button(
                "📥 Download Evaluation Excel",
                data=excel_buf,
                file_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
