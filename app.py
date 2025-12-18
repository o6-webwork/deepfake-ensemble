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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from shared_functions import (
    analyze_single_image,
    chat_with_model,
    load_ground_truth,
    calculate_metrics,
    display_confusion_matrix,
)
from config import PROMPTS, SYSTEM_PROMPT, MODEL_CONFIGS
from classifier import create_classifier_from_config
from detector import OSINTDetector
from spai_detector import SPAIDetector

st.set_page_config(page_title="NexInspect", layout="wide", page_icon="🔍")

# Cache SPAI model to avoid reloading on every inference (2GB+ model)
@st.cache_resource
def load_spai_detector():
    """
    Load and cache SPAI model (runs once per session).

    This is critical for performance - the SPAI model is 2GB+ and takes
    ~2 minutes to load from disk. Without caching, every analysis would
    reload the model.

    Returns:
        SPAIDetector instance (cached)
    """
    return SPAIDetector(
        config_path="spai/configs/spai.yaml",
        weights_path="spai/weights/spai.pth"
    )

# ==================== Analytics Functions ====================
# These functions support the Analytics tab

def load_evaluation_data_for_analytics(uploaded_file, config_name):
    """Load metrics from an evaluation Excel file."""
    try:
        metrics_df = pd.read_excel(uploaded_file, sheet_name='metrics')
        if len(metrics_df) == 0:
            return None

        row = metrics_df.iloc[0]
        data = {
            'Configuration': config_name,
            'Model': row.get('model', 'Unknown'),
            'Accuracy': row.get('accuracy', 0),
            'Precision': row.get('precision', 0),
            'Recall': row.get('recall', 0),
            'F1 Score': row.get('f1', 0),
            'TP': int(row.get('tp', 0)),
            'FP': int(row.get('fp', 0)),
            'TN': int(row.get('tn', 0)),
            'FN': int(row.get('fn', 0))
        }

        try:
            config_df = pd.read_excel(uploaded_file, sheet_name='config')
            for idx, config_row in config_df.iterrows():
                if config_row.get('parameter') == 'detection_mode':
                    data['Detection Mode'] = config_row.get('value', 'Unknown')
                    break
        except:
            data['Detection Mode'] = 'Unknown'

        return data
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")
        return None


def create_metrics_comparison(df):
    """Create bar chart comparing key metrics with dynamic sizing."""
    # Calculate dynamic height based on number of configs and label length
    n_configs = len(df)
    max_label_length = df['Configuration'].astype(str).str.len().max()

    # Dynamic spacing and sizing
    base_height = 600
    label_height_factor = min(max_label_length * 2, 150)  # Cap at 150px
    dynamic_height = base_height + label_height_factor

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for idx, (metric, color) in enumerate(zip(metrics, colors_list)):
        row = idx // 2 + 1
        col = idx % 2 + 1

        fig.add_trace(
            go.Bar(
                x=df['Configuration'],
                y=df[metric] * 100,
                name=metric,
                marker_color=color,
                text=df[metric].apply(lambda x: f'{x:.1%}'),
                textposition='outside',
                showlegend=False
            ),
            row=row, col=col
        )

        # Adjust tick angle based on label length
        tick_angle = -45 if max_label_length > 15 else 0

        fig.update_xaxes(
            tickangle=tick_angle,
            tickfont=dict(size=10),
            row=row, col=col
        )
        fig.update_yaxes(title_text="Percentage", range=[0, 105], row=row, col=col)

    # Dynamic bottom margin based on label length
    bottom_margin = max(80, min(label_height_factor, 180))

    fig.update_layout(
        height=dynamic_height,
        title_text="Performance Metrics Comparison",
        showlegend=False,
        margin=dict(b=bottom_margin, t=60, l=50, r=50)
    )

    return fig


def create_confusion_matrix_comparison(df):
    """Create confusion matrix visualization for each configuration."""
    n_configs = len(df)
    cols = min(3, n_configs)
    rows = (n_configs + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=df['Configuration'].tolist(),
        specs=[[{'type': 'heatmap'} for _ in range(cols)] for _ in range(rows)],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    for idx, row_data in df.iterrows():
        row_pos = idx // cols + 1
        col_pos = idx % cols + 1

        confusion = np.array([
            [row_data['TN'], row_data['FP']],
            [row_data['FN'], row_data['TP']]
        ])

        annotations = [
            [f"TN<br>{row_data['TN']}", f"FP<br>{row_data['FP']}"],
            [f"FN<br>{row_data['FN']}", f"TP<br>{row_data['TP']}"]
        ]

        fig.add_trace(
            go.Heatmap(
                z=confusion,
                x=['Predicted Real', 'Predicted Fake'],
                y=['Actual Real', 'Actual Fake'],
                text=annotations,
                texttemplate='%{text}',
                textfont={"size": 12},
                colorscale='Blues',
                showscale=(idx == 0),
                hovertemplate='%{text}<extra></extra>'
            ),
            row=row_pos, col=col_pos
        )

    fig.update_layout(
        height=300 * rows,
        title_text="Confusion Matrices"
    )

    return fig


def create_precision_recall_plot(df):
    """Create precision-recall scatter plot with smart label positioning."""
    # Calculate dynamic sizing based on number of configs
    n_configs = len(df)
    max_label_length = df['Configuration'].astype(str).str.len().max()

    # Increase height if many configs or long labels
    base_height = 500
    if n_configs > 5 or max_label_length > 20:
        base_height = 600

    fig = go.Figure()

    # Use smart text positioning to avoid overlap
    # For many configs, show labels on hover only
    if n_configs > 6:
        mode = 'markers'
        show_text = False
    else:
        mode = 'markers+text'
        show_text = True

    fig.add_trace(
        go.Scatter(
            x=df['Recall'] * 100,
            y=df['Precision'] * 100,
            mode=mode,
            text=df['Configuration'],
            textposition='top center',
            textfont=dict(size=10),
            marker=dict(
                size=15,
                color=df['F1 Score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="F1 Score"),
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>Precision: %{y:.1f}%<br>Recall: %{x:.1f}%<extra></extra>'
        )
    )

    fig.update_layout(
        title="Precision-Recall Trade-off",
        xaxis_title="Recall (%)",
        yaxis_title="Precision (%)",
        height=base_height,
        xaxis=dict(range=[0, 105]),
        yaxis=dict(range=[0, 105]),
        margin=dict(t=60, b=60, l=60, r=60)
    )

    return fig


def generate_pdf_report_analytics(df, metrics_comparison_fig, pr_plot_fig, confusion_matrix_fig):
    """Generate a comprehensive PDF report of the evaluation analytics."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)

    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12,
        spaceBefore=12
    )

    # Title
    title = Paragraph("NexInspect Evaluation Analytics Report", title_style)
    elements.append(title)

    timestamp = Paragraph(
        f"<para align=center>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</para>",
        styles['Normal']
    )
    elements.append(timestamp)
    elements.append(Spacer(1, 0.3*inch))

    # Summary Metrics Table
    elements.append(Paragraph("Summary Metrics", heading_style))

    table_data = [['Configuration', 'Accuracy', 'Precision', 'Recall', 'F1 Score']]
    for _, row in df.iterrows():
        table_data.append([
            str(row['Configuration']),
            f"{row['Accuracy']:.1%}",
            f"{row['Precision']:.1%}",
            f"{row['Recall']:.1%}",
            f"{row['F1 Score']:.1%}"
        ])

    table = Table(table_data, colWidths=[2.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))

    # Best Performers
    elements.append(Paragraph("Best Performers", heading_style))

    best_acc = df.loc[df['Accuracy'].idxmax()]
    best_f1 = df.loc[df['F1 Score'].idxmax()]

    best_text = f"""
    <b>Best Accuracy:</b> {best_acc['Configuration']} ({best_acc['Accuracy']:.1%})<br/>
    <b>Best F1 Score:</b> {best_f1['Configuration']} ({best_f1['F1 Score']:.1%})
    """

    elements.append(Paragraph(best_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    # Page break before charts
    elements.append(PageBreak())

    # Add charts as images
    elements.append(Paragraph("Performance Visualizations", heading_style))
    elements.append(Spacer(1, 0.2*inch))

    try:
        img_bytes = metrics_comparison_fig.to_image(format="png", width=700, height=500)
        img = RLImage(io.BytesIO(img_bytes), width=6*inch, height=4*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.2*inch))

        img_bytes = pr_plot_fig.to_image(format="png", width=700, height=500)
        img = RLImage(io.BytesIO(img_bytes), width=6*inch, height=4*inch)
        elements.append(img)
    except Exception as e:
        elements.append(Paragraph(f"<i>Note: Chart visualization requires kaleido package. Error: {str(e)}</i>", styles['Italic']))

    doc.build(elements)
    buffer.seek(0)

    return buffer

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
tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Evaluation", "📈 Analytics"])

# ───────────────────────── Tab 1: Single image ─────────────────────────
with tab1:
    st.header("💬 Analysis Chat")

    # Model configuration uploader (in sidebar or expander)
    with st.expander("⚙️ Model Configuration", expanded=False):
        st.markdown("Upload a `models.json` file to configure custom models and API keys.")
        uploaded_config = st.file_uploader(
            "Upload models.json",
            type=["json"],
            help="Upload a JSON file with model configurations. See models.json.example for template."
        )

        if uploaded_config is not None:
            try:
                import json
                import importlib
                from config import load_model_configs

                # Save uploaded file temporarily
                with open("models.json", "wb") as f:
                    f.write(uploaded_config.getbuffer())

                # Reload config module to pick up new configurations
                import config
                importlib.reload(config)

                # Update global MODEL_CONFIGS
                from config import MODEL_CONFIGS

                # Rebuild display mappings
                model_key_to_display = {
                    key: conf["display_name"] for key, conf in MODEL_CONFIGS.items()
                }
                display_to_model_key = {v: k for k, v in model_key_to_display.items()}

                st.success(f"✅ Loaded {len(MODEL_CONFIGS)} models from configuration file!")
                st.info("Models will be available after page refresh.")

            except Exception as e:
                st.error(f"❌ Error loading configuration: {str(e)}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download Template"):
                with open("models.json.example", "r") as f:
                    st.download_button(
                        label="Download models.json.example",
                        data=f.read(),
                        file_name="models.json",
                        mime="application/json"
                    )
        with col2:
            if st.button("🔄 Reload Models"):
                st.rerun()

    # Advanced Settings Expander (moved before model selector to get detection_mode early)
    with st.expander("⚙️ Advanced Settings", expanded=False):
        # Detection Mode Selector
        detection_mode = st.radio(
            "🔬 Detection Mode",
            options=["spai_assisted", "spai_standalone"],
            format_func=lambda x: {
                "spai_assisted": "SPAI + VLM (Comprehensive, ~3s)",
                "spai_standalone": "SPAI Only (Fast, ~50ms)"
            }[x],
            index=0,
            help="SPAI-assisted uses spectral analysis + VLM reasoning. SPAI-only is faster but less comprehensive."
        )

        # SPAI Configuration
        st.markdown("**SPAI Configuration**")

        spai_resolution = st.select_slider(
            "SPAI Resolution (Longest Edge)",
            options=[512, 768, 1024, 1280, 1536, 2048, "Original"],
            value=1280,
            help="Maximum resolution for SPAI spectral analysis. Higher = more accurate but slower."
        )

        spai_overlay_alpha = st.slider(
            "Heatmap Overlay Transparency",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Alpha blending: 0.0 = pure heatmap, 1.0 = pure original. Default 0.6 = 60% original + 40% heatmap."
        )

        # Watermark Mode Toggle (only for VLM mode)
        if detection_mode == "spai_assisted":
            watermark_mode = st.selectbox(
                "🏷️ Watermark Handling",
                options=["ignore", "analyze"],
                format_func=lambda x: {
                    "ignore": "Ignore (Treat as news logos)",
                    "analyze": "Analyze (Flag AI watermarks)"
                }[x],
                index=0,
                help="'Ignore' treats watermarks as news agency logos. 'Analyze' actively scans for AI tool watermarks (Sora, NanoBanana, etc.)"
            )
        else:
            watermark_mode = "ignore"
            st.info("💡 Watermark analysis requires VLM mode. Switch to 'SPAI + VLM' to enable.")

    # VLM model selector (disabled in standalone mode)
    vlm_disabled = (detection_mode == "spai_standalone")

    if vlm_disabled:
        st.selectbox(
            "Select detection model",
            ["(VLM disabled in SPAI standalone mode)"],
            index=0,
            disabled=True,
            help="VLM is not used in SPAI standalone mode. Switch to 'SPAI + VLM' to enable model selection."
        )
        detect_model_key = list(display_to_model_key.values())[0]  # Default (won't be used)
    else:
        detect_model_display = st.selectbox(
            "Select detection model",
            list(display_to_model_key.keys()),
            index=0,
        )
        detect_model_key = display_to_model_key[detect_model_display]

    # OSINT Context Selector (only for VLM mode)
    if detection_mode == "spai_assisted":
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
    else:
        osint_context = "auto"
        st.info("💡 OSINT context analysis requires VLM mode. SPAI standalone provides spectral analysis only.")

    # Debug Mode Toggle
    debug_mode = st.checkbox(
        "🔍 Enable Debug Mode",
        value=st.session_state.debug_mode,
        help="Show detailed forensic reports, VLM reasoning, and raw logprobs"
    )
    st.session_state.debug_mode = debug_mode

    # Analyze Button
    analyze_button = st.button(
        "🔍 Analyze Image",
        type="primary",
        disabled=(st.session_state.media is None),
        use_container_width=True,
        help="Run deepfake detection on the uploaded image"
    )

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
                # Trigger rerun to enable Analyze button
                st.rerun()
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
                        # Trigger rerun to enable Analyze button
                        st.rerun()
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

            # Display SPAI artifacts if available
            if st.session_state.forensic_artifacts is not None and st.session_state.forensic_artifacts[0] == "spai_heatmap":
                with st.expander("🔬 View SPAI Attention Heatmap", expanded=False):
                    st.markdown("### SPAI Spectral Analysis Overlay")

                    # Get heatmap from result
                    result = st.session_state.osint_result
                    if result and result.get('spai_heatmap_bytes'):
                        # Display the actual heatmap image
                        st.image(result['spai_heatmap_bytes'], use_container_width=True,
                                caption="SPAI Attention Heatmap (Blended Overlay)")

                        # Get SPAI scores if in debug mode
                        if 'debug' in result:
                            debug = result['debug']
                            spai_score = debug.get('spai_score', 0)
                            spai_pred = debug.get('spai_prediction', 'Unknown')

                            st.markdown(f"""
**SPAI Analysis:** {spai_pred} (Score: {spai_score:.3f})

**Heatmap Interpretation:**
- **Warm colors (Red/Orange/Yellow)**: Higher degree of AI manipulation detected
- **Dark Red**: Highest confidence of AI-generated artifacts
- **Cool colors (Blue/Purple)**: Normal spectral distributions (likely authentic)

The blended overlay (60% original + 40% heatmap) shows regions where SPAI's Vision Transformer
detected frequency-domain anomalies characteristic of AI generation. This heatmap was provided
to the VLM as visual context for comprehensive analysis.
                            """)
                        else:
                            st.markdown("""
**Heatmap Interpretation:**
- **Warm colors (Red/Orange/Yellow)**: Higher degree of AI manipulation detected
- **Dark Red**: Highest confidence of AI-generated artifacts
- **Cool colors (Blue/Purple)**: Normal spectral distributions (likely authentic)

Enable Debug Mode to see detailed SPAI scores.
                            """)
                    else:
                        st.info("SPAI heatmap is generated during analysis and provided to the VLM.")

                    # Show detailed OSINT result if available
                    if st.session_state.osint_result is not None:
                        result = st.session_state.osint_result
                        st.markdown("---")
                        st.markdown("### OSINT Detection Result")

                        tier = result['tier']
                        p_fake = result['confidence']  # detector.py returns P(fake) or "not available"

                        # Handle confidence display
                        if isinstance(p_fake, str):  # "not available"
                            confidence_display = "Not available (logprobs not supported)"
                            # Visual tier indicator without confidence bar
                            if tier == "Deepfake":
                                st.error(f"🚨 **{tier}** - AI Generated Probability: {confidence_display}")
                            elif tier == "Suspicious":
                                st.warning(f"⚠️ **{tier}** - AI Generated Probability: {confidence_display}")
                            else:
                                st.success(f"✅ **{tier}** - AI Generated Probability: {confidence_display}")
                        else:
                            # Visual confidence bar with color coding
                            if tier == "Deepfake":
                                st.error(f"🚨 **{tier}** - AI Generated Probability: {p_fake*100:.1f}%")
                            elif tier == "Suspicious":
                                st.warning(f"⚠️ **{tier}** - AI Generated Probability: {p_fake*100:.1f}%")
                            else:
                                st.success(f"✅ **{tier}** - AI Generated Probability: {p_fake*100:.1f}%")

                            st.progress(p_fake, text=f"AI Generated: {p_fake*100:.1f}%")

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

        for i, msg in enumerate(st.session_state.messages):
            role_class = "user" if msg["role"] == "user" else "assistant"

            # Check if this is an OSINT detection result (collapsible)
            if msg["role"] == "assistant" and msg.get("is_osint_result", False):
                # Extract filename and classification for expander title
                filename = msg.get("filename", "Unknown")
                tier = msg.get("tier", "Unknown")
                p_fake_pct = msg.get("p_fake_pct", None)

                # Determine emoji based on standardized three-tier system
                if tier == "Deepfake":
                    tier_emoji = "🚨"
                elif tier == "Suspicious":
                    tier_emoji = "⚠️"
                else:  # Authentic
                    tier_emoji = "✅"

                # Create collapsible expander with appropriate title
                if p_fake_pct is not None:
                    expander_title = f"{tier_emoji} **{filename}** - {tier} (AI Generated: {p_fake_pct:.1f}%)"
                else:
                    expander_title = f"{tier_emoji} **{filename}** - {tier} (Confidence: Not available)"

                with st.expander(
                    expander_title,
                    expanded=(i == len(st.session_state.messages) - 1)  # Only expand latest result
                ):
                    st.markdown(msg['content'], unsafe_allow_html=True)
            else:
                # Regular message (not collapsible)
                st.markdown(
                    f"<div class='chat-message {role_class}'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

        user_input = st.chat_input("Type a message...")

        st.markdown("</div>", unsafe_allow_html=True)

    if analysis_image is None:
        analysis_image = st.session_state.get("media")

    # Run analysis when Analyze button is clicked (not on upload)
    if analyze_button and analysis_image:
        with st.spinner("🔬 Running OSINT detection pipeline..."):
            try:
                # Add upload message if this is first analysis
                if uploaded_file and not any(
                    msg.get("content", "").startswith(f"📷 Uploaded image: {uploaded_file.name}")
                    or msg.get("content", "").startswith(f"🎞️ Uploaded video: {uploaded_file.name}")
                    for msg in st.session_state.messages
                ):
                    file_type = "📷 Uploaded image" if "image" in uploaded_file.type else "🎞️ Uploaded video"
                    st.session_state.messages.append(
                        {"role": "user", "content": f"{file_type}: {uploaded_file.name}"}
                    )
                # Convert PIL Image to bytes
                img_bytes = io.BytesIO()
                analysis_image.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()

                # Load cached SPAI detector (runs once per session)
                spai_detector = load_spai_detector()

                # Create OSINT detector with pre-loaded SPAI
                config = MODEL_CONFIGS[detect_model_key]
                detector = OSINTDetector(
                    base_url=config.get("base_url", ""),
                    model_name=config["model_name"],
                    api_key=config.get("api_key", "dummy"),
                    context=st.session_state.osint_context,
                    watermark_mode=watermark_mode,
                    provider=config.get("provider", "vllm"),
                    detection_mode=detection_mode,
                    spai_detector=spai_detector,  # Pass cached SPAI detector
                    spai_max_size=spai_resolution if spai_resolution != "Original" else None,
                    spai_overlay_alpha=spai_overlay_alpha
                )

                # Run detection with SPAI parameters
                import time
                detection_start = time.time()
                result = detector.detect(
                    img_bytes,
                    debug=st.session_state.debug_mode
                )
                detection_time = time.time() - detection_start

                # Store result
                st.session_state.osint_result = result

                # Store SPAI heatmap for display (always generated now)
                st.session_state.forensic_artifacts = None  # Clear old forensics
                if result.get('spai_heatmap_bytes'):
                    st.session_state.forensic_artifacts = ("spai_heatmap", None)  # Marker for SPAI mode

                # Create assistant message
                tier = result['tier']
                p_fake = result['confidence']  # detector.py returns P(fake) or "not available"

                # Handle confidence display
                if isinstance(p_fake, str):  # "not available"
                    p_fake_pct = None
                    confidence_display = "Not available (logprobs not supported)"
                else:
                    p_fake_pct = p_fake * 100
                    confidence_display = f"{p_fake_pct:.1f}%"

                reasoning = result['reasoning']

                # Determine color coding (standardized three-tier system)
                if tier == "Deepfake":
                    tier_emoji = "🚨"
                elif tier == "Suspicious":
                    tier_emoji = "⚠️"
                else:  # Authentic
                    tier_emoji = "✅"

                # Format mode description
                if detection_mode == "spai_standalone":
                    # SPAI standalone mode - show only SPAI results with clear prediction
                    # Note: SPAI always returns numeric scores, but add safety check
                    if isinstance(p_fake, (int, float)):
                        spai_prediction = "AI-Generated" if p_fake >= 0.5 else "Real"

                        # Show probability in clearer format
                        if spai_prediction == "AI-Generated":
                            prob_display = f"**AI-Generated Probability:** {p_fake_pct:.1f}%"
                        else:
                            prob_display = f"**Real Probability:** {(100 - p_fake_pct):.1f}% (AI: {p_fake_pct:.1f}%)"
                    else:
                        spai_prediction = "Unknown"
                        prob_display = f"**Confidence:** {confidence_display}"

                    assistant_msg = f"""**Detection Mode:** SPAI Standalone (Spectral Analysis Only)

**{tier_emoji} SPAI Prediction: {spai_prediction}**
{prob_display}

**Spectral Analysis:**
{reasoning}

💡 *View SPAI attention heatmap in the dropdown below the image*
"""
                else:
                    # SPAI + VLM assisted mode - show comprehensive analysis
                    assistant_msg = f"""**Model:** {detect_model_display}
**Detection Mode:** SPAI + VLM (Comprehensive Analysis)
**OSINT Context:** {osint_context.capitalize()}

**{tier_emoji} Classification: {tier}**
**AI Generated Probability:** {confidence_display}

**VLM Analysis:**
{reasoning}

**SPAI Spectral Report:**
```
{result.get('spai_report', 'No SPAI report available')}
```

💡 *View SPAI attention heatmap in the dropdown below the image*
"""

                # Add debug information if enabled
                if st.session_state.debug_mode and 'debug' in result:
                    debug = result['debug']

                    assistant_msg += f"""

---
### 🔬 Debug: SPAI Spectral Analysis (Raw Data)

**Detection Mode:** {debug.get('detection_mode', 'Unknown')}

**⏱️ Performance Timing:**
- Total Detection Time: {detection_time:.2f}s
- SPAI Analysis Time: {result.get('timing', {}).get('total', 0):.2f}s
- SPAI Device: {result.get('timing', {}).get('device', 'Unknown')}
{f"- VLM Request 1: {debug.get('request_1_latency', 0):.2f}s" if detection_mode == "spai_assisted" else ""}
{f"- VLM Request 2: {debug.get('request_2_latency', 0):.2f}s" if detection_mode == "spai_assisted" else ""}

**EXIF Metadata:**
```
{chr(10).join([f"{k}: {v}" for k, v in debug.get('exif_data', {}).items()]) if debug.get('exif_data') else '(No EXIF data found)'}
```

**SPAI Spectral Analysis:**
- SPAI Score: {debug.get('spai_score', 'N/A'):.4f} (AI generation probability)
- SPAI Prediction: {debug.get('spai_prediction', 'N/A')}
- SPAI Tier: {debug.get('spai_tier', 'N/A')}

**OSINT Context Applied:** {debug.get('context_applied', 'none').capitalize()}
"""
                    # Add VLM-specific debug info only if in assisted mode
                    if detection_mode == "spai_assisted":
                        assistant_msg += f"""
---
### 🧠 VLM Analysis Output

**Full Reasoning:**
{reasoning}

**API Metadata:**
- Model: {detect_model_display}
- Request 1 Latency: {debug.get('request_1_latency', 0):.2f}s
- Request 2 Latency: {debug.get('request_2_latency', 0):.2f}s (⚡ {((1 - debug.get('request_2_latency', 0)/max(debug.get('request_1_latency', 0.01), 0.01)) * 100):.1f}% faster via KV-cache)
- Request 1 Tokens: ~{debug.get('request_1_tokens', 0)}
- Request 2 Tokens: ~{debug.get('request_2_tokens', 0)}

---
### 📊 Logprobs & Verdict Extraction

**Top K=5 Tokens:**
"""
                        # Format top-k logprobs as table (only if available)
                        if debug.get('top_k_logprobs'):
                            for i, (token, logprob) in enumerate(debug.get('top_k_logprobs', [])[:5], 1):
                                prob = math.exp(logprob)
                                interpretation = ""
                                if token in detector.REAL_TOKENS:
                                    interpretation = "(REAL)"
                                elif token in detector.FAKE_TOKENS:
                                    interpretation = "(FAKE)"

                                assistant_msg += f"\n{i}. `{repr(token)}`: {logprob:.3f} → {prob:.4f} {interpretation}"
                        else:
                            assistant_msg += "\n*No logprobs available (not supported by this VLM provider)*"

                        # Show probabilities only if confidence is a number
                        if isinstance(p_fake, str):
                            assistant_msg += f"""

**Classification Result:**
- Verdict: {result.get('verdict_token', 'N/A')}
- Confidence: {confidence_display}
- Tier: **{tier}**

*Note: This VLM provider does not support logprobs. Classification is based on text-based A/B extraction.*

---
### ⚙️ System Prompt

```
{debug.get('system_prompt', 'N/A (standalone mode)')}
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
                        else:
                            assistant_msg += f"""

**Softmax Normalized Probabilities:**
- AI Generated: {p_fake:.4f} ({p_fake_pct:.1f}%)
- Authentic (Real): {(1 - p_fake):.4f} ({(1 - p_fake)*100:.1f}%)

**Three-Tier Classification Logic:**
- Tier: **{tier}**
- Threshold Check:
  * AI Generated < 50%? {'YES → Authentic' if p_fake < 0.50 else 'NO'}
  * AI Generated ≥ 90%? {'YES → Deepfake' if p_fake >= 0.90 else 'NO'}

**Verdict Token:** `{result.get('verdict_token', 'N/A')}`

---
### ⚙️ System Prompt

```
{debug.get('system_prompt', 'N/A (standalone mode)')}
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

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "is_osint_result": True,
                    "filename": uploaded_file.name if uploaded_file else "Unknown",
                    "tier": tier,
                    "p_fake_pct": p_fake_pct
                })
                st.rerun()

            except Exception as e:
                error_msg = f"❌ **Error during OSINT detection:**\n```\n{str(e)}\n```"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(f"Detection failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

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

    # Evaluation settings
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        eval_context = st.selectbox(
            "OSINT Context",
            options=["auto", "military", "disaster", "propaganda"],
            index=0,
            help="Context-specific detection protocols (CASE A/B/C)"
        )

    with col2:
        eval_detection_mode = st.radio(
            "Detection Mode",
            options=["spai_assisted", "spai_standalone"],
            format_func=lambda x: "SPAI + VLM" if x == "spai_assisted" else "SPAI Only",
            index=0,
            help="SPAI-assisted uses VLM reasoning, standalone is faster"
        )

    with col3:
        watermark_mode = st.selectbox(
            "Watermark Mode",
            options=["ignore", "analyze"],
            index=0,
            help="How to handle watermarks/logos"
        )

    with col4:
        eval_spai_resolution = st.select_slider(
            "SPAI Resolution",
            options=[512, 768, 1024, 1280, 1536, 2048, "Original"],
            value=1280,
            help="Maximum resolution for SPAI analysis (higher = more accurate but slower)"
        )

    # Choose which models to evaluate (disabled if SPAI standalone)
    eval_vlm_disabled = (eval_detection_mode == "spai_standalone")

    if eval_vlm_disabled:
        st.info("💡 VLM model selection is disabled in SPAI standalone mode. Only SPAI spectral analysis will be used.")
        models_to_run = [list(display_to_model_key.values())[0]]  # Default (won't be used)
    else:
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

            # Load cached SPAI detector ONCE for the entire evaluation
            spai_detector = load_spai_detector()

            per_model_results = {}  # model_key -> list[per image dicts]
            per_model_metrics = []  # list of metrics dicts with model info

            total_steps = len(eval_images) * len(models_to_run)
            progress_bar = st.progress(0)
            step = 0

            for model_key in models_to_run:
                model_config = MODEL_CONFIGS[model_key]
                model_display = model_config["display_name"]

                # Show appropriate header based on detection mode
                if eval_detection_mode == "spai_standalone":
                    st.write(f"### Running SPAI Standalone (Spectral Analysis Only)")
                else:
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

                    # Run detection once per image with cached SPAI detector
                    res = analyze_single_image(
                        image=img,
                        model_config=model_config,
                        context=eval_context,
                        watermark_mode=watermark_mode,
                        detection_mode=eval_detection_mode,
                        spai_max_size=eval_spai_resolution if eval_spai_resolution != "Original" else None,
                        spai_overlay_alpha=0.6,  # Default transparency
                        spai_detector=spai_detector  # Use cached SPAI detector
                    )

                    # Extract results
                    predicted_label = res["classification"]  # "Real" or "AI Generated"
                    confidence = res["confidence"]  # 0.0-1.0
                    tier = res["tier"]  # "Authentic" / "Suspicious" / "Deepfake"
                    analysis = res["analysis"]
                    verdict_token = res["verdict_token"]  # "A" or "B"

                    correct = predicted_label == actual

                    # Set model name appropriately for SPAI standalone vs assisted
                    if eval_detection_mode == "spai_standalone":
                        display_name = "SPAI Standalone (Spectral Analysis)"
                    else:
                        display_name = model_display

                    per_image_results.append(
                        {
                            "model_key": model_key,
                            "model_name": display_name,
                            "filename": filename,
                            "actual_label": actual,
                            "predicted_label": predicted_label,
                            "correct": correct,
                            "confidence": confidence,
                            "tier": tier,
                            "verdict_token": verdict_token,
                            "analysis": analysis,
                        }
                    )

                    step += 1
                    progress_bar.progress(step / total_steps)

                if not per_image_results:
                    st.warning(
                        f"No valid images with ground truth were processed for model {model_display}."
                    )
                    continue

                per_model_results[model_key] = per_image_results

                y_true = [r["actual_label"] for r in per_image_results]
                y_pred = [r["predicted_label"] for r in per_image_results]
                metrics = calculate_metrics(y_true, y_pred)
                metrics["model_key"] = model_key
                metrics["model_name"] = model_display
                per_model_metrics.append(metrics)

            if not per_model_metrics:
                st.error("No models produced valid results.")
                st.stop()

            # summary metrics table
            metrics_df = pd.DataFrame(per_model_metrics)
            st.subheader("📈 Evaluation Metrics by Model")
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

            st.subheader("📋 Per-Image Prediction Results (All Models)")
            df_display_ui = preds_df.copy()
            if "analysis" in df_display_ui.columns:
                df_display_ui["analysis_preview"] = (
                    df_display_ui["analysis"].str.slice(0, 100) + "..."
                )
            st.dataframe(
                df_display_ui[
                    [
                        "model_name",
                        "filename",
                        "actual_label",
                        "predicted_label",
                        "tier",
                        "confidence",
                        "verdict_token",
                        "correct",
                        "analysis_preview",
                    ]
                ]
            )

            # Excel export: config + metrics + predictions in 3 sheets
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                # Sheet 1: Evaluation configuration parameters
                config_data = {
                    "Parameter": [
                        "Detection Mode",
                        "OSINT Context",
                        "Watermark Mode",
                        "SPAI Resolution",
                        "SPAI Overlay Alpha",
                        "Timestamp",
                        "Total Images",
                        "Models Evaluated"
                    ],
                    "Value": [
                        eval_detection_mode,
                        eval_context,
                        watermark_mode,
                        str(eval_spai_resolution),
                        "0.6 (60% original + 40% heatmap)",
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        len(eval_images),
                        ", ".join([MODEL_CONFIGS[k]["display_name"] for k in models_to_run])
                    ]
                }
                config_df = pd.DataFrame(config_data)
                config_df.to_excel(writer, sheet_name="config", index=False)

                # Sheet 2: Metrics
                metrics_df.to_excel(writer, sheet_name="metrics", index=False)

                # Sheet 3: Predictions
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

# ───────────────────────── Tab 3: Analytics ─────────────────────────
with tab3:
    st.header("📈 Evaluation Analytics")
    st.markdown("Compare and analyze evaluation results with interactive visualizations and PDF export")

    # File uploader for evaluation results
    st.subheader("📁 Upload Evaluation Files")
    uploaded_analytics_files = st.file_uploader(
        "Select Excel evaluation files",
        type=['xlsx'],
        accept_multiple_files=True,
        help="Upload multiple evaluation result files for comparison",
        key="analytics_uploader"
    )

    if uploaded_analytics_files:
        # Configuration name input
        st.subheader("⚙️ Configuration Names")
        st.markdown("Assign names to each configuration:")

        config_names = {}
        cols = st.columns(min(3, len(uploaded_analytics_files)))
        for i, file in enumerate(uploaded_analytics_files):
            default_name = file.name.replace('evaluation_', '').replace('.xlsx', '')
            with cols[i % 3]:
                config_names[file.name] = st.text_input(
                    f"Config {i+1}:",
                    value=default_name,
                    key=f"analytics_name_{i}"
                )

        # Load all data
        all_data = []
        for file in uploaded_analytics_files:
            config_name = config_names.get(file.name, file.name)
            data = load_evaluation_data_for_analytics(file, config_name)
            if data:
                all_data.append(data)

        if all_data:
            df = pd.DataFrame(all_data)

            # Metrics table
            st.subheader("📋 Summary Metrics")
            display_df = df[['Configuration', 'Accuracy', 'Precision', 'Recall', 'F1 Score']].copy()
            display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f'{x:.1%}')
            display_df['Precision'] = display_df['Precision'].apply(lambda x: f'{x:.1%}')
            display_df['Recall'] = display_df['Recall'].apply(lambda x: f'{x:.1%}')
            display_df['F1 Score'] = display_df['F1 Score'].apply(lambda x: f'{x:.1%}')

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Download buttons
            col_csv, col_pdf = st.columns(2)

            with col_csv:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Comparison as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="evaluation_comparison.csv",
                    mime="text/csv"
                )

            # Visualizations
            st.subheader("📊 Performance Visualizations")

            # Metrics comparison
            metrics_fig = create_metrics_comparison(df)
            st.plotly_chart(metrics_fig, use_container_width=True)

            # Precision-Recall plot
            pr_fig = create_precision_recall_plot(df)
            st.plotly_chart(pr_fig, use_container_width=True)

            # Confusion matrices
            cm_fig = create_confusion_matrix_comparison(df)
            st.plotly_chart(cm_fig, use_container_width=True)

            # PDF Download button
            with col_pdf:
                try:
                    pdf_buffer = generate_pdf_report_analytics(df, metrics_fig, pr_fig, cm_fig)
                    st.download_button(
                        label="📄 Download Report as PDF",
                        data=pdf_buffer,
                        file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")
                    st.info("💡 Tip: Install kaleido for chart export: `pip install kaleido`")

            # Best Performers
            st.subheader("🏆 Best Performers")
            best_acc = df.loc[df['Accuracy'].idxmax()]
            best_f1 = df.loc[df['F1 Score'].idxmax()]
            best_precision = df.loc[df['Precision'].idxmax()]
            best_recall = df.loc[df['Recall'].idxmax()]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Accuracy", f"{best_acc['Accuracy']:.1%}", best_acc['Configuration'])
                st.metric("Best Precision", f"{best_precision['Precision']:.1%}", best_precision['Configuration'])
            with col2:
                st.metric("Best F1 Score", f"{best_f1['F1 Score']:.1%}", best_f1['Configuration'])
                st.metric("Best Recall", f"{best_recall['Recall']:.1%}", best_recall['Configuration'])

            # Dataset Composition
            st.subheader("🎯 Dataset Composition")
            total_samples = df.iloc[0]['TP'] + df.iloc[0]['FP'] + df.iloc[0]['TN'] + df.iloc[0]['FN']
            total_fakes = df.iloc[0]['TP'] + df.iloc[0]['FN']
            total_reals = df.iloc[0]['TN'] + df.iloc[0]['FP']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", total_samples)
            with col2:
                st.metric("Fake Images", f"{total_fakes} ({total_fakes/total_samples:.1%})")
            with col3:
                st.metric("Real Images", f"{total_reals} ({total_reals/total_samples:.1%})")

        else:
            st.warning("Could not load data from uploaded files. Please check file format.")
    else:
        st.info("👈 Upload evaluation files to begin analysis")

        st.markdown("""
        ### How to use this Analytics tab:

        1. **Run Evaluations:** First, use the **Evaluation** tab to run batch evaluations
        2. **Download Results:** Download the Excel files from completed evaluations
        3. **Upload Here:** Upload one or more Excel files using the uploader above
        4. **Compare:** View interactive charts and metrics comparisons
        5. **Export:** Download comparison data as CSV or comprehensive PDF report

        ### Supported Metrics:
        - Accuracy, Precision, Recall, F1 Score
        - Confusion Matrix (TP, FP, TN, FN)
        - Performance comparison charts
        - Precision-recall trade-off analysis

        ### Features:
        - 📊 Interactive Plotly visualizations
        - 📄 Professional PDF report generation
        - 📥 CSV export for further analysis
        - 🏆 Best performer identification
        """)
