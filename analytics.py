"""
Evaluation Results Analytics Tool

This script provides a comprehensive comparison of multiple evaluation results from
the deepfake detection system. It generates visualizations, statistical comparisons,
and detailed insights.

Usage:
    streamlit run analytics.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

st.set_page_config(page_title="NexInspect Analytics", layout="wide", page_icon="📊")

st.title("📊 NexInspect Evaluation Analytics")
st.markdown("Compare performance across multiple evaluation runs")

# File uploader
st.sidebar.header("Upload Evaluation Files")
uploaded_files = st.sidebar.file_uploader(
    "Select Excel evaluation files",
    type=['xlsx'],
    accept_multiple_files=True,
    help="Upload multiple evaluation result files for comparison"
)

# Configuration name input
if uploaded_files:
    st.sidebar.header("Configuration Names")
    st.sidebar.markdown("Assign names to each configuration:")

    config_names = {}
    for i, file in enumerate(uploaded_files):
        default_name = file.name.replace('evaluation_', '').replace('.xlsx', '')
        config_names[file.name] = st.sidebar.text_input(
            f"Config {i+1}:",
            value=default_name,
            key=f"name_{i}"
        )


def load_evaluation_data(uploaded_file, config_name):
    """Load metrics from an evaluation Excel file."""
    try:
        # Read metrics sheet
        metrics_df = pd.read_excel(uploaded_file, sheet_name='metrics')

        if len(metrics_df) == 0:
            return None

        row = metrics_df.iloc[0]

        # Extract metrics
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

        # Try to get detection mode from config sheet
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
    """Create bar chart comparing key metrics."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
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

        fig.update_yaxes(title_text="Percentage", range=[0, 105], row=row, col=col)

    fig.update_layout(
        height=600,
        title_text="Performance Metrics Comparison",
        showlegend=False
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

        # Confusion matrix: [[TN, FP], [FN, TP]]
        confusion = np.array([
            [row_data['TN'], row_data['FP']],
            [row_data['FN'], row_data['TP']]
        ])

        # Annotations
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


def create_improvement_chart(df, baseline_idx=0):
    """Create chart showing improvement from baseline."""
    baseline = df.iloc[baseline_idx]

    improvements = []
    for idx, row in df.iterrows():
        improvements.append({
            'Configuration': row['Configuration'],
            'Accuracy': (row['Accuracy'] - baseline['Accuracy']) * 100,
            'F1 Score': (row['F1 Score'] - baseline['F1 Score']) * 100,
            'Recall': (row['Recall'] - baseline['Recall']) * 100
        })

    imp_df = pd.DataFrame(improvements)

    fig = go.Figure()

    metrics = ['Accuracy', 'F1 Score', 'Recall']
    colors = ['#3498db', '#f39c12', '#2ecc71']

    for metric, color in zip(metrics, colors):
        fig.add_trace(
            go.Bar(
                name=metric,
                x=imp_df['Configuration'],
                y=imp_df[metric],
                text=imp_df[metric].apply(lambda x: f'{x:+.1f}%'),
                textposition='outside',
                marker_color=color
            )
        )

    fig.update_layout(
        title=f"Improvement from Baseline ({baseline['Configuration']})",
        xaxis_title="Configuration",
        yaxis_title="Percentage Point Change",
        barmode='group',
        height=400,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    )

    return fig


def create_precision_recall_plot(df):
    """Create precision-recall scatter plot."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df['Recall'] * 100,
            y=df['Precision'] * 100,
            mode='markers+text',
            text=df['Configuration'],
            textposition='top center',
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
        height=500,
        xaxis=dict(range=[0, 105]),
        yaxis=dict(range=[0, 105])
    )

    return fig


def generate_insights(df):
    """Generate textual insights from the data."""
    insights = []

    # Best performers
    best_acc = df.loc[df['Accuracy'].idxmax()]
    best_f1 = df.loc[df['F1 Score'].idxmax()]
    best_precision = df.loc[df['Precision'].idxmax()]
    best_recall = df.loc[df['Recall'].idxmax()]

    insights.append("### 🏆 Best Performers")
    insights.append(f"- **Best Accuracy:** {best_acc['Configuration']} ({best_acc['Accuracy']:.1%})")
    insights.append(f"- **Best F1 Score:** {best_f1['Configuration']} ({best_f1['F1 Score']:.1%})")
    insights.append(f"- **Best Precision:** {best_precision['Configuration']} ({best_precision['Precision']:.1%})")
    insights.append(f"- **Best Recall:** {best_recall['Configuration']} ({best_recall['Recall']:.1%})")
    insights.append("")

    # Range analysis
    insights.append("### 📊 Performance Range")
    insights.append(f"- **Accuracy Range:** {df['Accuracy'].min():.1%} - {df['Accuracy'].max():.1%} (Δ {(df['Accuracy'].max() - df['Accuracy'].min()):.1%})")
    insights.append(f"- **F1 Score Range:** {df['F1 Score'].min():.1%} - {df['F1 Score'].max():.1%} (Δ {(df['F1 Score'].max() - df['F1 Score'].min()):.1%})")
    insights.append("")

    # Dataset composition
    total_samples = df.iloc[0]['TP'] + df.iloc[0]['FP'] + df.iloc[0]['TN'] + df.iloc[0]['FN']
    total_fakes = df.iloc[0]['TP'] + df.iloc[0]['FN']
    total_reals = df.iloc[0]['TN'] + df.iloc[0]['FP']

    insights.append("### 🎯 Dataset Composition")
    insights.append(f"- **Total Samples:** {total_samples}")
    insights.append(f"- **Fake Images:** {total_fakes} ({total_fakes/total_samples:.1%})")
    insights.append(f"- **Real Images:** {total_reals} ({total_reals/total_samples:.1%})")
    insights.append("")

    # False positive/negative analysis
    insights.append("### ⚠️ Error Analysis")
    for _, row in df.iterrows():
        fp_rate = row['FP'] / (row['FP'] + row['TN']) if (row['FP'] + row['TN']) > 0 else 0
        fn_rate = row['FN'] / (row['FN'] + row['TP']) if (row['FN'] + row['TP']) > 0 else 0
        insights.append(f"**{row['Configuration']}:**")
        insights.append(f"  - False Positive Rate: {fp_rate:.1%} ({row['FP']} real images misclassified)")
        insights.append(f"  - False Negative Rate: {fn_rate:.1%} ({row['FN']} fake images missed)")
        insights.append("")

    return "\n".join(insights)


def load_predictions_data(uploaded_files, config_names):
    """Load per-image predictions from all uploaded files."""
    all_predictions = []

    for file in uploaded_files:
        config_name = config_names.get(file.name, file.name)
        try:
            # Read predictions sheet
            preds_df = pd.read_excel(file, sheet_name='predictions')

            # Add configuration name
            preds_df['configuration'] = config_name

            # Determine classification type (TP/TN/FP/FN)
            def classify_prediction(row):
                actual_fake = row['actual_label'] == 'AI Generated'
                pred_fake = row['predicted_label'] == 'AI Generated'

                if actual_fake and pred_fake:
                    return 'TP'
                elif not actual_fake and not pred_fake:
                    return 'TN'
                elif not actual_fake and pred_fake:
                    return 'FP'
                else:  # actual_fake and not pred_fake
                    return 'FN'

            preds_df['classification_type'] = preds_df.apply(classify_prediction, axis=1)

            all_predictions.append(preds_df)
        except Exception as e:
            st.warning(f"Could not load predictions from {file.name}: {e}")
            continue

    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
    return None


def display_prediction_card(row, index):
    """Display a single prediction as an expandable card."""
    # Determine color based on classification type
    colors = {
        'TP': '#28a745',  # Green
        'TN': '#17a2b8',  # Cyan
        'FP': '#dc3545',  # Red
        'FN': '#ffc107'   # Yellow
    }

    labels = {
        'TP': '✅ True Positive',
        'TN': '✅ True Negative',
        'FP': '❌ False Positive',
        'FN': '❌ False Negative'
    }

    color = colors.get(row['classification_type'], '#6c757d')
    label = labels.get(row['classification_type'], row['classification_type'])

    # Format confidence
    if isinstance(row.get('confidence'), str):
        conf_display = row['confidence']
    elif pd.notna(row.get('confidence')):
        conf_display = f"{row['confidence']:.1%}"
    else:
        conf_display = "N/A"

    # Create card header
    header = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <h4 style="margin: 0; color: white;">{label} - {row['filename']}</h4>
        <p style="margin: 5px 0 0 0; color: white; font-size: 0.9em;">
            Model: {row.get('model_name', 'Unknown')} |
            Tier: {row.get('tier', 'N/A')} |
            Confidence: {conf_display} |
            Token: {row.get('verdict_token', 'N/A')}
        </p>
    </div>
    """

    st.markdown(header, unsafe_allow_html=True)

    # Show details in columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Actual Label:** {row['actual_label']}")
        st.markdown(f"**Configuration:** {row.get('configuration', 'N/A')}")
    with col2:
        st.markdown(f"**Predicted Label:** {row['predicted_label']}")
        st.markdown(f"**Correct:** {'✅ Yes' if row.get('correct', False) else '❌ No'}")

    # Display full analysis in expandable section
    if pd.notna(row.get('analysis')) and row['analysis']:
        with st.expander("📄 View Full Model Response", expanded=False):
            st.text_area(
                "Analysis",
                value=row['analysis'],
                height=400,
                key=f"analysis_{index}_{row['filename']}",
                label_visibility="collapsed"
            )
    else:
        st.info("No analysis text available for this prediction.")

    st.markdown("---")


def generate_pdf_report(df, metrics_comparison_fig, pr_plot_fig, confusion_matrix_fig):
    """
    Generate a comprehensive PDF report of the evaluation analytics.

    Args:
        df: DataFrame with evaluation metrics
        metrics_comparison_fig: Plotly figure for metrics comparison
        pr_plot_fig: Plotly figure for precision-recall plot
        confusion_matrix_fig: Plotly figure for confusion matrices

    Returns:
        BytesIO buffer containing the PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)

    # Container for the 'Flowable' objects
    elements = []

    # Define styles
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

    # Timestamp
    timestamp = Paragraph(
        f"<para align=center>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</para>",
        styles['Normal']
    )
    elements.append(timestamp)
    elements.append(Spacer(1, 0.3*inch))

    # Summary Metrics Table
    elements.append(Paragraph("Summary Metrics", heading_style))

    # Prepare table data
    table_data = [['Configuration', 'Accuracy', 'Precision', 'Recall', 'F1 Score']]
    for _, row in df.iterrows():
        table_data.append([
            str(row['Configuration']),
            f"{row['Accuracy']:.1%}",
            f"{row['Precision']:.1%}",
            f"{row['Recall']:.1%}",
            f"{row['F1 Score']:.1%}"
        ])

    # Create table
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

    # Confusion Matrix Data
    elements.append(Paragraph("Confusion Matrix Data", heading_style))

    cm_table_data = [['Configuration', 'TP', 'FP', 'TN', 'FN']]
    for _, row in df.iterrows():
        cm_table_data.append([
            str(row['Configuration']),
            str(row['TP']),
            str(row['FP']),
            str(row['TN']),
            str(row['FN'])
        ])

    cm_table = Table(cm_table_data, colWidths=[3*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    cm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
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

    elements.append(cm_table)
    elements.append(Spacer(1, 0.3*inch))

    # Best Performers
    elements.append(Paragraph("Best Performers", heading_style))

    best_acc = df.loc[df['Accuracy'].idxmax()]
    best_f1 = df.loc[df['F1 Score'].idxmax()]
    best_precision = df.loc[df['Precision'].idxmax()]
    best_recall = df.loc[df['Recall'].idxmax()]

    best_text = f"""
    <b>Best Accuracy:</b> {best_acc['Configuration']} ({best_acc['Accuracy']:.1%})<br/>
    <b>Best F1 Score:</b> {best_f1['Configuration']} ({best_f1['F1 Score']:.1%})<br/>
    <b>Best Precision:</b> {best_precision['Configuration']} ({best_precision['Precision']:.1%})<br/>
    <b>Best Recall:</b> {best_recall['Configuration']} ({best_recall['Recall']:.1%})
    """

    elements.append(Paragraph(best_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    # Dataset Composition
    elements.append(Paragraph("Dataset Composition", heading_style))

    total_samples = df.iloc[0]['TP'] + df.iloc[0]['FP'] + df.iloc[0]['TN'] + df.iloc[0]['FN']
    total_fakes = df.iloc[0]['TP'] + df.iloc[0]['FN']
    total_reals = df.iloc[0]['TN'] + df.iloc[0]['FP']

    dataset_text = f"""
    <b>Total Samples:</b> {total_samples}<br/>
    <b>Fake Images:</b> {total_fakes} ({total_fakes/total_samples:.1%})<br/>
    <b>Real Images:</b> {total_reals} ({total_reals/total_samples:.1%})
    """

    elements.append(Paragraph(dataset_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    # Page break before charts
    elements.append(PageBreak())

    # Add charts as images
    elements.append(Paragraph("Performance Visualizations", heading_style))
    elements.append(Spacer(1, 0.2*inch))

    # Convert Plotly figures to images
    try:
        # Metrics comparison chart
        img_bytes = metrics_comparison_fig.to_image(format="png", width=700, height=500)
        img = Image(io.BytesIO(img_bytes), width=6*inch, height=4*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.2*inch))

        # Precision-Recall plot
        img_bytes = pr_plot_fig.to_image(format="png", width=700, height=500)
        img = Image(io.BytesIO(img_bytes), width=6*inch, height=4*inch)
        elements.append(img)

    except Exception as e:
        elements.append(Paragraph(f"<i>Note: Chart visualization requires kaleido package for export. Error: {str(e)}</i>", styles['Italic']))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    return buffer


# Main application
if uploaded_files:
    # Load all data
    all_data = []
    for file in uploaded_files:
        config_name = config_names.get(file.name, file.name)
        data = load_evaluation_data(file, config_name)
        if data:
            all_data.append(data)

    if all_data:
        df = pd.DataFrame(all_data)

        # Load predictions data for the new tab
        predictions_df = load_predictions_data(uploaded_files, config_names)

        # Create tabs
        tab1, tab2 = st.tabs(["📊 Overview & Metrics", "🔍 Prediction Viewer"])

        # ==================== Tab 1: Overview & Metrics ====================
        with tab1:
            # Metrics table
            st.header("📋 Summary Metrics")
            display_df = df[['Configuration', 'Accuracy', 'Precision', 'Recall', 'F1 Score']].copy()
            display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f'{x:.1%}')
            display_df['Precision'] = display_df['Precision'].apply(lambda x: f'{x:.1%}')
            display_df['Recall'] = display_df['Recall'].apply(lambda x: f'{x:.1%}')
            display_df['F1 Score'] = display_df['F1 Score'].apply(lambda x: f'{x:.1%}')

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Download buttons for CSV and PDF export
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
            st.header("📊 Visualizations")

            # Metrics comparison
            metrics_fig = create_metrics_comparison(df)
            st.plotly_chart(metrics_fig, use_container_width=True)

            # Precision-Recall plot
            pr_fig = create_precision_recall_plot(df)
            st.plotly_chart(pr_fig, use_container_width=True)

            # Improvement chart (if multiple configs)
            if len(df) > 1:
                baseline_options = df['Configuration'].tolist()
                baseline_selection = st.selectbox(
                    "Select baseline for improvement comparison:",
                    baseline_options,
                    index=0
                )
                baseline_idx = df[df['Configuration'] == baseline_selection].index[0]
                st.plotly_chart(create_improvement_chart(df, baseline_idx), use_container_width=True)

            # Confusion matrices
            cm_fig = create_confusion_matrix_comparison(df)
            st.plotly_chart(cm_fig, use_container_width=True)

            # PDF Download button (after all charts are created)
            with col_pdf:
                try:
                    pdf_buffer = generate_pdf_report(df, metrics_fig, pr_fig, cm_fig)
                    st.download_button(
                        label="📄 Download Report as PDF",
                        data=pdf_buffer,
                        file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")
                    st.info("💡 Tip: Install kaleido for chart export: `pip install kaleido`")

            # Insights
            st.header("💡 Insights")
            st.markdown(generate_insights(df))

            # Detailed breakdown
            st.header("🔍 Detailed Breakdown")
            with st.expander("View Full Data Table"):
                st.dataframe(df, use_container_width=True, hide_index=True)

        # ==================== Tab 2: Prediction Viewer ====================
        with tab2:
            if predictions_df is not None and len(predictions_df) > 0:
                st.header("🔍 Prediction Viewer")
                st.markdown("Filter and explore individual predictions with full model responses")

                # Filtering sidebar
                st.subheader("🎯 Filters")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Filter by classification type
                    classification_types = ['All', 'TP', 'TN', 'FP', 'FN']
                    selected_type = st.selectbox(
                        "Classification Result",
                        classification_types,
                        help="TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative"
                    )

                with col2:
                    # Filter by configuration
                    configs = ['All'] + sorted(predictions_df['configuration'].unique().tolist())
                    selected_config = st.selectbox("Configuration", configs)

                with col3:
                    # Filter by correctness
                    correctness_options = ['All', 'Correct', 'Incorrect']
                    selected_correctness = st.selectbox("Prediction Correctness", correctness_options)

                # Additional filters
                col4, col5, col6 = st.columns(3)

                with col4:
                    # Filter by tier
                    if 'tier' in predictions_df.columns:
                        tiers = ['All'] + sorted([t for t in predictions_df['tier'].unique() if pd.notna(t)])
                        selected_tier = st.selectbox("Tier", tiers)
                    else:
                        selected_tier = 'All'

                with col5:
                    # Filter by verdict token
                    if 'verdict_token' in predictions_df.columns:
                        tokens = ['All'] + sorted([t for t in predictions_df['verdict_token'].unique() if pd.notna(t)])
                        selected_token = st.selectbox("Verdict Token", tokens)
                    else:
                        selected_token = 'All'

                with col6:
                    # Filter by model
                    if 'model_name' in predictions_df.columns:
                        models = ['All'] + sorted(predictions_df['model_name'].unique().tolist())
                        selected_model = st.selectbox("Model", models)
                    else:
                        selected_model = 'All'

                # Apply filters
                filtered_df = predictions_df.copy()

                if selected_type != 'All':
                    filtered_df = filtered_df[filtered_df['classification_type'] == selected_type]

                if selected_config != 'All':
                    filtered_df = filtered_df[filtered_df['configuration'] == selected_config]

                if selected_correctness != 'All':
                    if selected_correctness == 'Correct':
                        filtered_df = filtered_df[filtered_df['correct'] == True]
                    else:
                        filtered_df = filtered_df[filtered_df['correct'] == False]

                if selected_tier != 'All':
                    filtered_df = filtered_df[filtered_df['tier'] == selected_tier]

                if selected_token != 'All':
                    filtered_df = filtered_df[filtered_df['verdict_token'] == selected_token]

                if selected_model != 'All':
                    filtered_df = filtered_df[filtered_df['model_name'] == selected_model]

                # Statistics panel
                st.markdown("---")
                st.subheader("📈 Filtered Results Statistics")

                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

                with stat_col1:
                    st.metric("Total Predictions", len(filtered_df))

                with stat_col2:
                    if len(filtered_df) > 0:
                        correct_count = filtered_df['correct'].sum()
                        accuracy = correct_count / len(filtered_df)
                        st.metric("Accuracy", f"{accuracy:.1%}")
                    else:
                        st.metric("Accuracy", "N/A")

                with stat_col3:
                    # Average confidence (if numeric)
                    if 'confidence' in filtered_df.columns and len(filtered_df) > 0:
                        numeric_conf = pd.to_numeric(filtered_df['confidence'], errors='coerce')
                        if numeric_conf.notna().any():
                            avg_conf = numeric_conf.mean()
                            st.metric("Avg Confidence", f"{avg_conf:.1%}")
                        else:
                            st.metric("Avg Confidence", "N/A")
                    else:
                        st.metric("Avg Confidence", "N/A")

                with stat_col4:
                    # Distribution
                    if len(filtered_df) > 0:
                        tp_count = (filtered_df['classification_type'] == 'TP').sum()
                        tn_count = (filtered_df['classification_type'] == 'TN').sum()
                        fp_count = (filtered_df['classification_type'] == 'FP').sum()
                        fn_count = (filtered_df['classification_type'] == 'FN').sum()
                        st.metric("TP/TN/FP/FN", f"{tp_count}/{tn_count}/{fp_count}/{fn_count}")
                    else:
                        st.metric("TP/TN/FP/FN", "0/0/0/0")

                # Export filtered results
                if len(filtered_df) > 0:
                    csv_export = io.StringIO()
                    filtered_df.to_csv(csv_export, index=False)
                    st.download_button(
                        label="📥 Download Filtered Results as CSV",
                        data=csv_export.getvalue(),
                        file_name=f"filtered_predictions_{selected_type}.csv",
                        mime="text/csv"
                    )

                st.markdown("---")

                # Display predictions
                if len(filtered_df) > 0:
                    st.subheader(f"📋 Showing {len(filtered_df)} Prediction(s)")

                    # Sort options
                    sort_col1, sort_col2 = st.columns([3, 1])
                    with sort_col1:
                        sort_by = st.selectbox(
                            "Sort by",
                            ['filename', 'confidence', 'tier', 'classification_type'],
                            index=0
                        )
                    with sort_col2:
                        sort_order = st.radio("Order", ['Ascending', 'Descending'], horizontal=True)

                    # Sort dataframe
                    ascending = (sort_order == 'Ascending')
                    filtered_df_sorted = filtered_df.sort_values(by=sort_by, ascending=ascending)

                    # Display each prediction as a card
                    for idx, row in filtered_df_sorted.iterrows():
                        display_prediction_card(row, idx)

                else:
                    st.info("No predictions match the selected filters.")

            else:
                st.warning("No prediction data available. The uploaded files may not contain a 'predictions' sheet.")

    else:
        st.warning("Could not load data from uploaded files. Please check file format.")
else:
    st.info("👈 Upload evaluation files using the sidebar to begin analysis")

    st.markdown("""
    ### How to use this tool:

    1. **Upload Files:** Use the sidebar to upload multiple evaluation result Excel files
    2. **Name Configurations:** Assign meaningful names to each configuration
    3. **Analyze:** View automated comparison charts and insights
    4. **Export:** Download comparison data as CSV for further analysis

    ### Supported Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix (TP, FP, TN, FN)

    ### Visualizations:
    - Performance metrics comparison
    - Precision-recall trade-off plot
    - Improvement from baseline
    - Confusion matrix heatmaps
    - Statistical insights
    """)
