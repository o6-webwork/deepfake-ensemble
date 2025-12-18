# Evaluation Results Summary

**Generated:** 2025-12-16
**Dataset:** 72 images (48 Fake + 24 Real)

## Performance Comparison

| Configuration | Accuracy | Precision | Recall | F1 Score | TP | FP | TN | FN |
|--------------|----------|-----------|--------|----------|----|----|----|----|
| Old System (Qwen) | 55.6% | 90.0% | 37.5% | 52.9% | 18 | 2 | 22 | 30 |
| New VLM Only (Qwen) | 61.1% | 88.5% | 47.9% | 62.2% | 23 | 3 | 21 | 25 |
| SPAI Standalone | 69.4% | 90.6% | 60.4% | 72.5% | 29 | 3 | 21 | 19 |
| SPAI + VLM (Qwen) | 63.9% | 92.3% | 50.0% | 64.9% | 24 | 2 | 22 | 24 |

## Improvement from Baseline

| Configuration | Accuracy Δ | F1 Score Δ | Recall Δ |
|--------------|-----------|-----------|----------|
| Old System (Qwen) | +0.0% | +0.0% | +0.0% |
| New VLM Only (Qwen) | +5.5% | +9.2% | +10.4% |
| SPAI Standalone | +13.9% | +19.6% | +22.9% |
| SPAI + VLM (Qwen) | +8.3% | +11.9% | +12.5% |

## Key Findings

### 🏆 Best Performers

- **Best Overall (Accuracy & F1):** SPAI Standalone
  - 69.4% accuracy (+13.9% vs baseline)
  - 72.5% F1 score (+19.6% vs baseline)
  - Caught 29/48 fakes (60.4% recall)

- **Most Precise:** SPAI + VLM
  - 92.3% precision (only 2 false positives)
  - Best for scenarios where false alarms are costly

- **Best Recall:** SPAI Standalone
  - 60.4% recall (caught most fakes)
  - Missed 19/48 fakes (compared to 30/48 in baseline)

### 📊 Analysis

**Prompt Engineering Impact:**
- New prompt alone (VLM Only) improved accuracy by 5.5% and F1 by 9.2%
- Shows effective prompt refinement

**SPAI Impact:**
- SPAI standalone delivers best overall performance
- SPAI + VLM combination has highest precision but lower recall
- Spectral analysis proves highly effective for this dataset

**Surprising Result:**
- SPAI standalone outperforms SPAI + VLM hybrid
- Suggests VLM may be overriding correct SPAI predictions
- May indicate need for prompt tuning in SPAI-assisted mode

### ⚠️ Error Analysis

**Old System (Qwen):**
- False Positive Rate: 8.3% (2 real images misclassified)
- False Negative Rate: 62.5% (30 fake images missed)
- High precision, very low recall

**New VLM Only (Qwen):**
- False Positive Rate: 12.5% (3 real images misclassified)
- False Negative Rate: 52.1% (25 fake images missed)
- Improved recall, slightly lower precision

**SPAI Standalone:**
- False Positive Rate: 12.5% (3 real images misclassified)
- False Negative Rate: 39.6% (19 fake images missed)
- Best balance of precision and recall

**SPAI + VLM (Qwen):**
- False Positive Rate: 8.3% (2 real images misclassified)
- False Negative Rate: 50.0% (24 fake images missed)
- Highest precision but conservative

## Recommendations

### 1. Make SPAI Standalone the Default Mode
**Rationale:**
- Highest overall performance (69.4% accuracy)
- Best F1 score (72.5%)
- Faster inference (~5s vs ~8s)
- Simpler pipeline

**Use cases:**
- Batch processing
- Real-time analysis
- High-throughput scenarios

### 2. Use SPAI + VLM for High-Stakes Scenarios
**Rationale:**
- Highest precision (92.3%)
- Lowest false positive rate
- More conservative predictions

**Use cases:**
- Legal evidence
- Forensic investigations
- When false positives have serious consequences

### 3. Investigate Hybrid System Underperformance
**Possible causes:**
- VLM overriding correct SPAI predictions
- SPAI-assisted prompt needs tuning
- Test dataset may favor spectral over semantic analysis

**Suggested actions:**
- Review cases where SPAI + VLM disagrees with SPAI alone
- A/B test different prompt strategies for SPAI integration
- Evaluate on diverse datasets to confirm findings

### 4. Address Recall Challenge
**Current state:**
- All systems struggle with recall (missing 39-63% of fakes)
- Even best system (SPAI) misses 19/48 fakes

**Potential improvements:**
- Lower decision thresholds for higher recall
- Ensemble methods combining multiple models
- Additional training data for edge cases
- User-configurable sensitivity settings

## Dataset Characteristics

- **Total samples:** 72 images
- **Fake images:** 48 (66.7%)
- **Real images:** 24 (33.3%)
- **Class imbalance:** 2:1 fake-to-real ratio

This imbalance means accuracy alone can be misleading. F1 score provides a better balanced metric.

## Visualizations Available

The following visualizations are available in the generated reports:

1. **Performance Metrics Comparison** - 4-panel bar chart
2. **Precision-Recall Trade-off** - Scatter plot showing balance
3. **Improvement from Baseline** - Grouped bar chart
4. **Confusion Matrices** - Heatmap grid for all configurations

Generate interactive visualizations with:
```bash
streamlit run analytics.py
```

Or static HTML report with:
```bash
python generate_analytics_report.py results/*.xlsx
```
