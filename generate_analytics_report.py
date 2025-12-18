#!/usr/bin/env python3
"""
Generate Analytics Report (CLI Version)

This script generates a comprehensive analytics report comparing multiple evaluation
results, saving visualizations as HTML files.

Usage:
    python generate_analytics_report.py result1.xlsx result2.xlsx result3.xlsx --output report.html
    python generate_analytics_report.py results/*.xlsx --labels "Old System" "New System" "SPAI"
"""

import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path


def load_evaluation_data(filepath, config_name):
    """Load metrics from an evaluation Excel file."""
    try:
        metrics_df = pd.read_excel(filepath, sheet_name='metrics')

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

        # Try to get detection mode
        try:
            config_df = pd.read_excel(filepath, sheet_name='config')
            for idx, config_row in config_df.iterrows():
                if config_row.get('parameter') == 'detection_mode':
                    data['Detection Mode'] = config_row.get('value', 'Unknown')
                    break
        except:
            data['Detection Mode'] = 'Unknown'

        return data

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
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
    """Create confusion matrix visualization."""
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


def create_improvement_chart(df, baseline_idx=0):
    """Create improvement chart from baseline."""
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


def generate_html_report(df, output_file):
    """Generate comprehensive HTML report."""

    # Create all figures
    metrics_fig = create_metrics_comparison(df)
    pr_fig = create_precision_recall_plot(df)
    improvement_fig = create_improvement_chart(df, 0) if len(df) > 1 else None
    confusion_fig = create_confusion_matrix_comparison(df)

    # Generate insights
    best_acc = df.loc[df['Accuracy'].idxmax()]
    best_f1 = df.loc[df['F1 Score'].idxmax()]
    best_precision = df.loc[df['Precision'].idxmax()]
    best_recall = df.loc[df['Recall'].idxmax()]

    total_samples = df.iloc[0]['TP'] + df.iloc[0]['FP'] + df.iloc[0]['TN'] + df.iloc[0]['FN']
    total_fakes = df.iloc[0]['TP'] + df.iloc[0]['FN']
    total_reals = df.iloc[0]['TN'] + df.iloc[0]['FP']

    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NexInspect Evaluation Analytics</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                margin: 20px;
                background: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 8px;
            }}
            h3 {{
                color: #7f8c8d;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ecf0f1;
            }}
            th {{
                background: #3498db;
                color: white;
                font-weight: 600;
            }}
            tr:hover {{
                background: #f8f9fa;
            }}
            .insight {{
                background: #e8f4f8;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 15px 0;
                border-radius: 4px;
            }}
            .metric {{
                display: inline-block;
                margin: 10px 20px 10px 0;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
            }}
            .metric-label {{
                font-size: 12px;
                color: #7f8c8d;
                text-transform: uppercase;
            }}
            .plot {{
                margin: 30px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 NexInspect Evaluation Analytics</h1>
            <p><em>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>

            <h2>Summary Metrics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Configuration</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>TP</th>
                        <th>FP</th>
                        <th>TN</th>
                        <th>FN</th>
                    </tr>
                </thead>
                <tbody>
    """

    for _, row in df.iterrows():
        html += f"""
                    <tr>
                        <td><strong>{row['Configuration']}</strong></td>
                        <td>{row['Accuracy']:.1%}</td>
                        <td>{row['Precision']:.1%}</td>
                        <td>{row['Recall']:.1%}</td>
                        <td>{row['F1 Score']:.1%}</td>
                        <td>{row['TP']}</td>
                        <td>{row['FP']}</td>
                        <td>{row['TN']}</td>
                        <td>{row['FN']}</td>
                    </tr>
        """

    html += f"""
                </tbody>
            </table>

            <h2>Key Insights</h2>
            <div class="insight">
                <h3>🏆 Best Performers</h3>
                <div class="metric">
                    <div class="metric-label">Best Accuracy</div>
                    <div class="metric-value">{best_acc['Accuracy']:.1%}</div>
                    <div>{best_acc['Configuration']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Best F1 Score</div>
                    <div class="metric-value">{best_f1['F1 Score']:.1%}</div>
                    <div>{best_f1['Configuration']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Best Precision</div>
                    <div class="metric-value">{best_precision['Precision']:.1%}</div>
                    <div>{best_precision['Configuration']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Best Recall</div>
                    <div class="metric-value">{best_recall['Recall']:.1%}</div>
                    <div>{best_recall['Configuration']}</div>
                </div>
            </div>

            <div class="insight">
                <h3>🎯 Dataset Composition</h3>
                <p><strong>Total Samples:</strong> {total_samples}</p>
                <p><strong>Fake Images:</strong> {total_fakes} ({total_fakes/total_samples:.1%})</p>
                <p><strong>Real Images:</strong> {total_reals} ({total_reals/total_samples:.1%})</p>
            </div>

            <h2>Visualizations</h2>

            <div class="plot" id="metrics-chart"></div>
            <div class="plot" id="pr-chart"></div>
    """

    if improvement_fig:
        html += '<div class="plot" id="improvement-chart"></div>'

    html += """
            <div class="plot" id="confusion-chart"></div>

            <script>
    """

    # Add Plotly figures
    html += f"Plotly.newPlot('metrics-chart', {metrics_fig.to_json()});\n"
    html += f"Plotly.newPlot('pr-chart', {pr_fig.to_json()});\n"

    if improvement_fig:
        html += f"Plotly.newPlot('improvement-chart', {improvement_fig.to_json()});\n"

    html += f"Plotly.newPlot('confusion-chart', {confusion_fig.to_json()});\n"

    html += """
            </script>
        </div>
    </body>
    </html>
    """

    # Write to file
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate analytics report from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare multiple evaluation files
  python generate_analytics_report.py result1.xlsx result2.xlsx result3.xlsx

  # Use custom labels
  python generate_analytics_report.py *.xlsx --labels "Baseline" "V2" "V3"

  # Custom output file
  python generate_analytics_report.py *.xlsx --output custom_report.html
        """
    )

    parser.add_argument('files', nargs='+', help='Evaluation Excel files to compare')
    parser.add_argument('--labels', nargs='+', help='Custom labels for each file')
    parser.add_argument('--output', '-o', default='analytics_report.html',
                        help='Output HTML file (default: analytics_report.html)')

    args = parser.parse_args()

    # Load all data
    all_data = []
    for i, filepath in enumerate(args.files):
        # Determine config name
        if args.labels and i < len(args.labels):
            config_name = args.labels[i]
        else:
            config_name = Path(filepath).stem.replace('evaluation_', '')

        data = load_evaluation_data(filepath, config_name)
        if data:
            all_data.append(data)
            print(f"Loaded: {filepath} as '{config_name}'")

    if not all_data:
        print("Error: No valid evaluation data found")
        return 1

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Generate report
    generate_html_report(df, args.output)

    # Also save CSV
    csv_file = args.output.replace('.html', '.csv')
    df.to_csv(csv_file, index=False)
    print(f"CSV data saved: {csv_file}")

    return 0


if __name__ == '__main__':
    exit(main())
