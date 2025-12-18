# Evaluation Analytics Tools

This directory contains tools for analyzing and comparing evaluation results from the NexInspect deepfake detection system.

## Tools

### 1. Interactive Analytics Dashboard (analytics.py)

A Streamlit-based web interface for interactive analysis of evaluation results.

**Features:**
- Upload multiple evaluation Excel files
- Interactive visualizations (bar charts, precision-recall plots, confusion matrices)
- Real-time comparison with configurable baseline
- Statistical insights generation
- Export comparison data as CSV

**Usage:**
```bash
streamlit run analytics.py
```

Then open your browser to http://localhost:8501 and upload evaluation files using the sidebar.

**Screenshots:**
- Performance metrics comparison (4-panel bar chart)
- Precision-recall trade-off scatter plot
- Improvement from baseline (grouped bar chart)
- Confusion matrix heatmaps
- Automated insights panel

---

### 2. CLI Report Generator (generate_analytics_report.py)

A command-line tool for generating static HTML reports comparing multiple evaluation runs.

**Features:**
- Batch processing of evaluation files
- Generates standalone HTML reports with embedded visualizations
- Exports comparison data to CSV
- Custom labels for each configuration

**Usage:**

Basic usage (auto-generates labels from filenames):
```bash
python generate_analytics_report.py results/eval1.xlsx results/eval2.xlsx results/eval3.xlsx
```

With custom labels:
```bash
python generate_analytics_report.py \
  results/eval1.xlsx results/eval2.xlsx results/eval3.xlsx \
  --labels "Baseline" "SPAI Only" "SPAI + VLM"
```

With custom output location:
```bash
python generate_analytics_report.py results/*.xlsx --output reports/comparison.html
```

Using wildcards:
```bash
python generate_analytics_report.py results/evaluation_*.xlsx
```

**Output:**
- `analytics_report.html` - Standalone HTML report (can be opened in any browser)
- `analytics_report.csv` - Raw comparison data

---

## Example Workflow

### Comparing System Versions

```bash
# Generate comprehensive report comparing all system versions
python generate_analytics_report.py \
  "results/evaluation_20251202_213404_migrated.xlsx" \
  "results/evaluation_20251215_144928.xlsx" \
  "results/evaluation_20251216_041250.xlsx" \
  "results/evaluation_20251216_095143.xlsx" \
  --labels "Old System (Qwen)" "New VLM Only" "SPAI Standalone" "SPAI + VLM" \
  --output "results/system_comparison.html"

# Open the report
xdg-open results/system_comparison.html
```

### Interactive Exploration

```bash
# Launch interactive dashboard
streamlit run analytics.py

# Then:
# 1. Upload evaluation files via sidebar
# 2. Name each configuration
# 3. Explore visualizations
# 4. Adjust baseline for improvement calculations
# 5. Download CSV for further analysis
```

---

## Visualizations

### 1. Performance Metrics Comparison
4-panel bar chart showing:
- Accuracy
- Precision
- Recall
- F1 Score

Each metric displayed as a percentage with values labeled on bars.

### 2. Precision-Recall Trade-off
Scatter plot showing the balance between precision and recall across configurations. Points are:
- Positioned by Precision (Y) and Recall (X)
- Colored by F1 Score (darker = higher)
- Labeled with configuration name

Useful for understanding whether a system is conservative (high precision, low recall) or aggressive (high recall, low precision).

### 3. Improvement from Baseline
Grouped bar chart showing percentage point changes in:
- Accuracy
- F1 Score
- Recall

Relative to a selected baseline configuration. Helps visualize the impact of system changes.

### 4. Confusion Matrices
Heatmap grid showing the confusion matrix for each configuration:
- **TN (True Negative)**: Real images correctly identified
- **FP (False Positive)**: Real images misclassified as fake
- **FN (False Negative)**: Fake images missed
- **TP (True Positive)**: Fake images correctly detected

---

## Metrics Explained

### Accuracy
`(TP + TN) / Total`

Overall correctness. Proportion of all predictions that were correct.

### Precision
`TP / (TP + FP)`

When the system says "fake", how often is it correct? High precision = few false alarms.

### Recall (Sensitivity)
`TP / (TP + FN)`

Of all actual fakes, how many did we catch? High recall = few missed fakes.

### F1 Score
`2 × (Precision × Recall) / (Precision + Recall)`

Harmonic mean of precision and recall. Balanced metric that considers both false positives and false negatives.

---

## Insights Generated

The tools automatically generate insights including:

1. **Best Performers**: Which configuration achieved the highest score for each metric
2. **Performance Range**: Spread between best and worst performers
3. **Dataset Composition**: Number of fake vs real images in test set
4. **Error Analysis**: False positive and false negative rates for each configuration
5. **Trade-off Analysis**: Which systems prioritize precision vs recall

---

## Requirements

```bash
pip install streamlit pandas plotly openpyxl numpy
```

All dependencies are included in the main `requirements.txt`.

---

## File Format Requirements

Evaluation Excel files must contain a `metrics` sheet with the following columns:
- `model`: Model name (optional)
- `accuracy`: Accuracy (0.0-1.0)
- `precision`: Precision (0.0-1.0)
- `recall`: Recall (0.0-1.0)
- `f1`: F1 Score (0.0-1.0)
- `tp`: True Positives (integer)
- `fp`: False Positives (integer)
- `tn`: True Negatives (integer)
- `fn`: False Negatives (integer)

Optional `config` sheet with `parameter` and `value` columns for additional metadata.

This format is automatically generated by the evaluation feature in the main application.

---

## Tips

1. **Use descriptive labels**: Clear configuration names make reports easier to interpret
2. **Compare incrementally**: Start with baseline vs. one change to isolate impact
3. **Check confusion matrices**: Raw TP/FP/TN/FN counts reveal patterns that percentages might hide
4. **Look at both precision AND recall**: High scores in only one suggest the system is imbalanced
5. **Validate on multiple datasets**: Performance on one test set may not generalize

---

## Troubleshooting

**Error: "Worksheet named 'metrics' not found"**
- Ensure you're uploading evaluation result files from the NexInspect system
- Check that the Excel file contains a sheet named exactly "metrics"

**Plotly visualizations not displaying**
- Ensure you have an active internet connection (for loading Plotly.js in HTML reports)
- Try opening the HTML report in a different browser

**Streamlit app won't start**
- Check if port 8501 is already in use: `lsof -i :8501`
- Specify a different port: `streamlit run analytics.py --server.port 8502`

**Labels not appearing correctly**
- Ensure number of labels matches number of files
- Use quotes for labels with spaces: `--labels "System A" "System B"`
