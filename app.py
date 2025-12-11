import streamlit as st
from PIL import Image
from collections import Counter
import io
import tempfile
import cv2
import pandas as pd
import os
import math
from datetime import datetime

from shared_functions import (
    analyze_single_image,
    chat_with_model,
    load_ground_truth,
    calculate_metrics,
    display_confusion_matrix,
)
from config import PROMPTS, SYSTEM_PROMPT, MODEL_CONFIGS
from forensics import generate_both, ArtifactGenerator
from classifier import create_classifier_from_config
from detector import OSINTDetector

st.set_page_config(page_title="Deepfake Detector", layout="wide", page_icon="🕵️‍♂️")

# session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "media" not in st.session_state:
    st.session_state.media = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = []
if "forensic_artifacts" not in st.session_state:
    st.session_state.forensic_artifacts = None  # Stores (ela_bytes, fft_bytes)
if "forensic_result" not in st.session_state:
    st.session_state.forensic_result = None  # Stores classification result (legacy)
if "osint_context" not in st.session_state:
    st.session_state.osint_context = "auto"  # auto/military/disaster/propaganda
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "osint_result" not in st.session_state:
    st.session_state.osint_result = None  # Stores OSINTDetector result

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

    # OSINT Context Selector
    osint_context = st.selectbox(
        "OSINT Context",
        options=["auto", "military", "disaster", "propaganda"],
        format_func=lambda x: {
            "auto": "Auto-Detect",
            "military": "Military (Uniforms/Parades/Formations)",
            "disaster": "Disaster/HADR (Flood/Rubble/Combat)",
            "propaganda": "Propaganda/Showcase (Studio/News)"
        }[x],
        index=0,
        help="Select scene type for context-adaptive forensic thresholds"
    )
    st.session_state.osint_context = osint_context

    # Debug Mode Toggle
    debug_mode = st.checkbox(
        "🔍 Enable Debug Mode",
        value=st.session_state.debug_mode,
        help="Show detailed forensic reports, VLM reasoning, and raw logprobs"
    )
    st.session_state.debug_mode = debug_mode

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
                # Video processing with proper cleanup
                tmp_path = None
                try:
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
                finally:
                    # Cleanup temporary file
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass  # Ignore cleanup errors

    # media panel
    with left_col:
        st.markdown('<div id="media-panel">', unsafe_allow_html=True)
        if st.session_state.media is not None:
            st.image(st.session_state.media, use_container_width=True, caption="Original Image")

            # Display forensic artifacts if available
            if st.session_state.forensic_artifacts is not None:
                with st.expander("🔬 View Forensic Artifacts", expanded=False):
                    ela_bytes, fft_bytes = st.session_state.forensic_artifacts

                    # Three-column layout for artifacts
                    st.markdown("### Forensic Analysis Maps")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**ELA (Error Level Analysis)**")
                        st.image(
                            Image.open(io.BytesIO(ela_bytes)),
                            use_container_width=True,
                            caption="Compression inconsistency map"
                        )
                        st.caption("🔍 **AI signature:** Uniform rainbow static\n\n✓ **Real signature:** Dark regions with edge noise")

                    with col2:
                        st.markdown("**FFT (Frequency Spectrum)**")
                        st.image(
                            Image.open(io.BytesIO(fft_bytes)),
                            use_container_width=True,
                            caption="Frequency domain analysis"
                        )
                        st.caption("🔍 **AI signature:** Grid/cross patterns\n\n✓ **Real signature:** Chaotic starburst")

                    # Show detailed OSINT result if available
                    if st.session_state.osint_result is not None:
                        result = st.session_state.osint_result
                        st.markdown("---")
                        st.markdown("### OSINT Detection Result")

                        tier = result['tier']
                        confidence = result['confidence']

                        # Visual confidence bar with color coding
                        if tier == "Deepfake":
                            st.error(f"🚨 **{tier}** - Confidence: {confidence*100:.1f}%")
                        elif tier == "Suspicious":
                            st.warning(f"⚠️ **{tier}** - Confidence: {confidence*100:.1f}%")
                        else:
                            st.success(f"✅ **{tier}** - Confidence: {confidence*100:.1f}%")

                        st.progress(confidence, text=f"P(Fake): {confidence*100:.1f}%")

                        # Metadata auto-fail indicator
                        if result.get('metadata_auto_fail', False):
                            st.error("⚠️ AI tool signature detected in metadata - Instant rejection")

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
        with st.spinner("🔬 Running OSINT detection pipeline..."):
            try:
                # Convert PIL Image to bytes
                img_bytes = io.BytesIO()
                analysis_image.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()

                # Create OSINT detector
                config = MODEL_CONFIGS[detect_model_key]
                detector = OSINTDetector(
                    base_url=config["base_url"],
                    model_name=config["model_name"],
                    api_key=config.get("api_key", "dummy"),
                    context=st.session_state.osint_context
                )

                # Run detection with debug mode
                result = detector.detect(
                    img_bytes,
                    debug=st.session_state.debug_mode
                )

                # Store result
                st.session_state.osint_result = result

                # Generate artifacts for display
                ag = ArtifactGenerator()
                ela_bytes = ag.generate_ela(img_bytes)
                fft_bytes, _ = ag.generate_fft_preprocessed(img_bytes)
                st.session_state.forensic_artifacts = (ela_bytes, fft_bytes)

                # Create assistant message
                tier = result['tier']
                confidence_pct = result['confidence'] * 100
                reasoning = result['reasoning']

                # Determine color coding
                if tier == "Deepfake":
                    tier_emoji = "🚨"
                elif tier == "Suspicious":
                    tier_emoji = "⚠️"
                else:
                    tier_emoji = "✅"

                assistant_msg = f"""**Model:** {detect_model_display}
**OSINT Context:** {osint_context.capitalize()}

**{tier_emoji} Classification: {tier}**
**Confidence:** {confidence_pct:.1f}%

**VLM Reasoning:**
{reasoning}

**Forensic Report:**
```
{result['forensic_report']}
```

💡 *View forensic artifacts below the image panel*
"""

                # Add debug information if enabled
                if st.session_state.debug_mode and 'debug' in result:
                    debug = result['debug']

                    assistant_msg += f"""

---
### 🔬 Debug: Forensic Lab Report (Raw Data)

**EXIF Metadata:**
```
{chr(10).join([f"{k}: {v}" for k, v in debug['exif_data'].items()]) if debug['exif_data'] else '(No EXIF data found)'}
```

**ELA Analysis:**
- Variance Score: {debug['ela_variance']:.2f}
- Threshold: <2.0 (AI indicator)

**FFT Analysis:**
- Pattern Type: {debug['fft_pattern']}
- Peaks Detected: {debug['fft_peaks']}

**OSINT Context Applied:** {debug['context_applied'].capitalize()}

---
### 🧠 VLM Analysis Output

**Full Reasoning:**
{reasoning}

**API Metadata:**
- Model: {detect_model_display}
- Request 1 Latency: {debug['request_1_latency']:.2f}s
- Request 2 Latency: {debug['request_2_latency']:.2f}s (⚡ {((1 - debug['request_2_latency']/max(debug['request_1_latency'], 0.01)) * 100):.1f}% faster via KV-cache)
- Request 1 Tokens: ~{debug['request_1_tokens']}
- Request 2 Tokens: ~{debug['request_2_tokens']}

---
### 📊 Logprobs & Verdict Extraction

**Top K=5 Tokens:**
"""
                    # Format top-k logprobs as table
                    for i, (token, logprob) in enumerate(debug['top_k_logprobs'][:5], 1):
                        prob = math.exp(logprob)
                        interpretation = ""
                        if token in detector.REAL_TOKENS:
                            interpretation = "(REAL)"
                        elif token in detector.FAKE_TOKENS:
                            interpretation = "(FAKE)"

                        assistant_msg += f"\n{i}. `{repr(token)}`: {logprob:.3f} → {prob:.4f} {interpretation}"

                    assistant_msg += f"""

**Softmax Normalized:**
- P(Fake) = {result['confidence']:.4f} ({confidence_pct:.1f}%)
- P(Real) = {(1 - result['confidence']):.4f} ({(1 - result['confidence'])*100:.1f}%)

**Three-Tier Classification:**
- Tier: **{tier}**
- Threshold Check:
  * P_fake < 0.50? {'YES → Authentic' if result['confidence'] < 0.50 else 'NO'}
  * P_fake ≥ 0.90? {'YES → Deepfake' if result['confidence'] >= 0.90 else 'NO'}

**Verdict Token:** `{result['verdict_token']}`

---
### ⚙️ System Prompt

```
{debug['system_prompt']}
```

---
### ⏱️ Performance Metrics

**Stage-by-Stage Timing:**
- Stage 0 (Metadata): {debug.get('stage_0_time', 0):.3f}s
- Stage 1 (Forensics): {debug.get('stage_1_time', 0):.3f}s
- Stage 2 (VLM Analysis): {debug['request_1_latency']:.2f}s
- Stage 3 (Verdict): {debug['request_2_latency']:.2f}s

**Total Pipeline:** {debug['total_pipeline_time']:.2f}s

**KV-Cache Hit:** {'✅ YES' if debug.get('kv_cache_hit', False) else '❌ NO'}
"""

                st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
                st.rerun()

            except Exception as e:
                error_msg = f"❌ **Error during OSINT detection:**\n```\n{str(e)}\n```"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(f"Detection failed: {str(e)}")

                # Fallback to old method
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

            # Optimize ground truth lookup by converting to set (O(1) instead of O(N))
            gt_filenames = set(gt_df["filename"].values)

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

                    if filename not in gt_filenames:
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
