# Analysis Output Guide

## Generated Files Overview

All analysis outputs have been generated and saved in this directory. Below is a complete guide to using these files for presentations, reports, or further analysis.

---

## 📄 Reports

### **COMPREHENSIVE_REPORT.md** (19 KB)
**Purpose:** Complete written analysis including methodology, findings, and recommendations

**Contains:**
- Executive summary
- Detailed evaluation methodology with prompts used
- Complete dataset breakdown
- Model-by-model performance analysis
- AIG vs AIM comparison insights
- Confusion matrix definitions in context
- Recommendations for future work
- All metrics formulas and explanations

**Best for:**
- Reading comprehensive findings
- Understanding methodology
- Getting context for all visualizations
- Academic or technical documentation

---

### **dataset_summary.txt** (606 bytes)
**Purpose:** Quick reference for dataset statistics

**Contains:**
- Total image counts
- Category distribution
- Scenario breakdown
- Time of day split

**Best for:**
- Quick reference during presentations
- Slide deck annotations
- Dataset description slides

---

### **detailed_results.csv** (567 bytes)
**Purpose:** Raw data for custom analysis

**Contains:**
- All metrics for each model in CSV format
- Can be imported into Excel, Python, R, etc.

**Columns:**
- Model, Overall Accuracy, Precision, Recall, F1 Score
- TP, TN, FP, FN
- AIG Recall, AIG Detected, AIG Total
- AIM Recall, AIM Detected, AIM Total
- Real Specificity, False Positives

**Best for:**
- Creating custom visualizations
- Statistical analysis
- Importing into other tools

---

## 📊 Visualizations (Ready for Slide Decks)

All images are high-resolution (300 DPI) PNG files suitable for presentations and publications.

---

### **1_model_comparison.png** (444 KB)
**4-panel comprehensive comparison**

**Panels:**
1. **Top-left:** Overall performance metrics (Accuracy, Precision, Recall, F1) bar chart
2. **Top-right:** AIG vs AIM detection rates comparison
3. **Bottom-left:** Confusion matrix for best performing model (InternVL 3.5 8B)
4. **Bottom-right:** Performance by category breakdown

**Use in slides:**
- Overview slide showing all key metrics
- Model comparison section
- Can be split into 4 separate slides if needed

**Key insight shown:** InternVL 3.5 8B has the best balance across all metrics

---

### **2_confusion_matrices.png** (199 KB)
**Side-by-side confusion matrices with percentages**

**Shows:** All three models' confusion matrices with:
- Absolute counts
- Percentage of total
- Color-coded by performance (green = good, red = poor)

**Use in slides:**
- Detailed performance comparison
- Explaining TP/TN/FP/FN for each model
- Showing trade-offs between models

**Key insight shown:** InternVL 2.5 and MiniCPM have 0 false positives but massive false negatives

---

### **3_dataset_composition.png** (239 KB)
**3-panel dataset breakdown**

**Panels:**
1. **Left:** Pie chart of category distribution (AIG/AIM/REAL)
2. **Center:** Horizontal bar chart of disaster scenarios
3. **Right:** Day vs Night distribution

**Use in slides:**
- Dataset description slide
- Methodology section
- Showing balanced evaluation approach

**Key insight shown:** Perfectly balanced 33/33/33 split ensures fair comparison

---

### **4_performance_table.png** (147 KB)
**Summary table as image**

**Shows:** Professional formatted table with:
- All models and their key metrics
- AIG and AIM recall rates
- TP/TN/FP/FN counts
- Color-coded rows for readability

**Use in slides:**
- Results summary slide
- Quick reference table
- Executive summary presentations

**Key insight shown:** Complete performance overview in one glance

---

### **5_detection_rates.png** (168 KB)
**Large bar chart comparing detection rates**

**Shows:** Side-by-side comparison of:
- AIG Detection Rate (orange)
- AIM Detection Rate (green)
- Real Identification Rate (blue)

**Use in slides:**
- Highlighting AIG vs AIM difficulty
- Showing model strengths/weaknesses
- Main findings slide

**Key insight shown:** AIM is consistently harder to detect than AIG

---

### **6_confusion_definitions.png** (622 KB)
**Educational slide explaining confusion matrix**

**Shows:** Full text explanation of:
- True Positive (TP)
- True Negative (TN)
- False Positive (FP) - with warnings
- False Negative (FN) - with warnings
- Trade-off analysis
- Use-case specific guidance

**Use in slides:**
- Background/methodology section
- Explaining metrics to non-technical audience
- Context for why different metrics matter
- Training materials

**Key insight shown:** What each metric means in real-world disaster response context

---

## 🎯 Suggested Slide Deck Structure

### **For Technical Audience (10-15 slides):**

1. **Title Slide**
2. **Agenda/Outline**
3. **Background** - Why deepfake detection matters in HADR
4. **Methodology** - Use dataset_summary.txt + 3_dataset_composition.png
5. **Evaluation Approach** - Include prompts from COMPREHENSIVE_REPORT.md
6. **Confusion Matrix Primer** - Use 6_confusion_definitions.png
7. **Overall Results** - Use 4_performance_table.png
8. **Model Comparison** - Use 1_model_comparison.png (or split into 4 slides)
9. **Key Finding: AIG vs AIM** - Use 5_detection_rates.png
10. **Detailed Confusion Matrices** - Use 2_confusion_matrices.png
11. **Model-by-Model Analysis** - Text from COMPREHENSIVE_REPORT.md with charts
12. **Insights & Implications**
13. **Recommendations**
14. **Q&A**

---

### **For Executive/Non-Technical Audience (6-8 slides):**

1. **Title Slide**
2. **The Problem** - Why we need deepfake detection
3. **Our Evaluation** - Use 3_dataset_composition.png (show balanced dataset)
4. **Key Results** - Use 4_performance_table.png (simplified explanation)
5. **Main Finding** - Use 5_detection_rates.png (explain AIM harder than AIG)
6. **Recommendation** - Which model to use and why
7. **Next Steps**
8. **Q&A**

---

## 💡 Key Messages for Presentation

### **Main Findings:**
1. ✅ **InternVL 3.5 8B is the best overall model** (48.61% accuracy, best F1 score)
2. ⚠️ **AI-Manipulated images are harder to detect** than fully AI-generated (8% difference)
3. 🎯 **Trade-off exists:** High precision models miss most fakes, balanced models have more false alarms
4. 📊 **Even best model misses >50% of fakes** - human verification still critical

### **Recommendations:**
- **Social Media Platforms** → Use InternVL 3.5 8B (catch more fakes, accept some false alarms)
- **Emergency Response** → Use MiniCPM-V 4.5 (zero false alarms, but needs other verification)
- **All Applications** → Implement multi-stage verification with human review

---

## 🔄 Regenerating Analysis

To regenerate this analysis with updated data:

```bash
cd "/home/otb-02/Desktop/deepfake detection"
python3 generate_report.py
```

This will:
- Read the latest evaluation results
- Regenerate all visualizations
- Update all reports
- Overwrite files in analysis_output/

---

## 📧 Citation

When using these materials in publications or presentations:

```
Deepfake Detection Evaluation Report
Vision-Language Model Performance Analysis
Generated: December 8, 2025
Models Evaluated: InternVL 2.5 8B, InternVL 3.5 8B, MiniCPM-V 4.5
Dataset: 72 images (24 REAL, 24 AIG, 24 AIM) across HADR scenarios
```

---

## ❓ Quick Reference

**Need a quick stat?** → dataset_summary.txt
**Need all metrics in Excel?** → detailed_results.csv
**Need to understand everything?** → COMPREHENSIVE_REPORT.md
**Need slide-ready graphics?** → All PNG files
**Need to explain confusion matrix?** → 6_confusion_definitions.png
**Need to show best model?** → 1_model_comparison.png
**Need to show AIG vs AIM difference?** → 5_detection_rates.png

---

**Questions? Check COMPREHENSIVE_REPORT.md for detailed explanations of all findings.**
