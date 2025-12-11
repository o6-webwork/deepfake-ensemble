# Executive Summary: Deepfake Detection Evaluation

**Date:** December 8, 2025
**Models Tested:** 4 Vision-Language Models
**Dataset:** 72 images across 3 domains (HADR, Military Conflict, Military Showcase)
**Composition:** 24 Real, 24 AI-Generated, 24 AI-Manipulated

---

## 🎯 Bottom Line

**Qwen3 VL 32B is the best model** for deepfake detection across critical scenarios, achieving 55.56% accuracy with excellent precision (90%) and only 2 false positives out of 72 images.

However, **even the best model misses >62% of fake images**, highlighting the absolute need for human verification in critical applications involving disaster response or military intelligence.

---

## 📊 Key Findings

### 1. Model Rankings

| Rank | Model | Accuracy | Precision | Recall | Best For |
|------|-------|----------|-----------|--------|----------|
| 🥇 | **Qwen3 VL 32B** | 55.56% | 90.00% | 37.50% | Balanced detection, military intelligence |
| 🥈 | **InternVL 3.5 8B** | 48.61% | 68.97% | 41.67% | Social media moderation, high recall |
| 🥉 | **MiniCPM-V 4.5** | 41.67% | 100.00% | 12.50% | Emergency response (zero false alarms) |
| 4 | **InternVL 2.5 8B** | 34.72% | 100.00% | 2.08% | Not recommended |

### 2. The AIG vs AIM Challenge

**Major Discovery:** AI-Manipulated images (real photos edited with AI) are **consistently ~8% harder to detect** than fully AI-Generated images across all capable models.

| Image Type | Qwen3 VL | InternVL 3.5 | MiniCPM-V |
|------------|----------|--------------|-----------|
| **AI-Generated (AIG)** | 41.7% | 45.8% | 16.7% |
| **AI-Manipulated (AIM)** | 33.3% ⚠️ | 37.5% ⚠️ | 8.3% ⚠️ |
| **Real Images (Specificity)** | 91.7% ✅ | 62.5% | 100.0% ✅ |

**Why?** Manipulated images retain authentic characteristics (real lighting, textures, physics) that confuse detection models, while fully generated images have more detectable artifacts throughout.

### 3. The Precision-Recall Trade-off

**Qwen3 VL 32B (Best Overall):**
- Excellent precision (90%) - only 2 false positives ✅
- Catches 37.5% of fakes (18/48) ⚠️
- Misses 62.5% of fakes (30/48) 🚨
- **Use case:** General-purpose detection, military intelligence, news verification

**InternVL 3.5 8B (Highest Recall):**
- Catches 41.67% of fakes (highest) ✅
- But wrongly flags 37.5% of real images as fake ⚠️
- **Use case:** Social media moderation where catching fakes is priority

**MiniCPM-V 4.5 (Zero False Alarms):**
- Perfect precision (100%) - never wrong when it flags ✅
- But misses 87.5% of fakes 🚨
- **Use case:** Emergency response where false alarms are unacceptable

---

## 🗺️ Dataset Composition

### Three Domains (Equal Balance)

**HADR (Humanitarian Assistance & Disaster Response)** - 24 images
- Earthquakes, fires, floods, storms
- Critical for emergency response decision-making
- False positives can delay real disaster response

**Military Conflict** - 24 images
- Airstrikes, ground battles, convoys, urban damage
- High-stakes intelligence applications
- False negatives can lead to misinformation campaigns

**Military Showcase** - 24 images
- Parades, marching, live fire exercises, navigation
- Public relations and documentation
- Reputation-critical applications

**Perfect Balance:** Each domain contains 8 AIG + 8 AIM + 8 REAL images

### Time Distribution
- **50% Day images** (36)
- **50% Night images** (36)

---

## 🔍 What Was Evaluated

### Prompt Used:
Models were asked to score images 0-10 (0=real, 10=fake) based on:
- Semantic coherence
- Lighting, textures, artifacts
- Signs of digital editing

**Threshold:** Score > 4 → Classified as "AI Generated"

### Evaluation Method:
- **5 runs per image** per model with majority vote (consensus)
- **4 models tested:** Qwen3 VL 32B, InternVL 3.5 8B, MiniCPM-V 4.5, InternVL 2.5 8B
- **72 unique images** = 360 total evaluations per model

---

## ⚠️ What This Means in Practice

### False Positives (FP) - "Crying Wolf"
Real disaster/military photos wrongly flagged as fake.
- **Impact:** Undermines trust, causes people to ignore warnings
- **Example:** Actual military convoy photo marked as AI-generated, intelligence officers doubt its authenticity
- **Qwen3 VL:** Only 2 FP (2.8% of dataset) ✅

### False Negatives (FN) - "Missing the Threat"
Fake images passing as real.
- **Impact:** Misinformation spreads, causes panic, misdirects resources
- **Example:** AI-generated disaster scene goes viral, triggering unnecessary evacuations
- **Best model still misses:** 62.5% of fakes 🚨

### The Balance:
- **Military Intelligence:** Prioritize low FP (credibility critical) → Use Qwen3 VL
- **Social Media:** Prioritize low FN (can't allow viral fakes) → Use InternVL 3.5
- **Emergency Ops:** Zero FP required (can't miss real disasters) → Use MiniCPM-V
- **News/Journalism:** Balance both with mandatory human review → Use Qwen3 VL + human verification

---

## 💡 Recommendations

### Immediate Actions:

1. **Deploy Qwen3 VL 32B** as primary detection system
   - Best overall performance (55.56% accuracy)
   - Excellent precision (90%) with only 2 false positives
   - Suitable for automated screening in critical applications

2. **Always include human verification** for high-stakes decisions
   - No model is reliable enough for autonomous decisions
   - Especially critical for military intelligence and emergency response

3. **Implement multi-stage verification pipeline:**
   ```
   Stage 1: Automated AI Detection (Qwen3 VL)
       ↓
   Stage 2: Reverse Image Search
       ↓
   Stage 3: Human Expert Review (for flagged content)
       ↓
   Stage 4: Contextual Verification (source, timing, consistency)
   ```

4. **Use InternVL 3.5 as secondary check** for high-threat scenarios
   - Highest recall (41.67%)
   - Good for catching fakes that Qwen3 VL might miss
   - Accept higher false positive rate with human review

### Future Improvements:

1. **Threshold Optimization:**
   - Test thresholds from 3.0 to 5.0 (currently using 4.0)
   - Create ROC curves for optimal operating points
   - Develop domain-specific thresholds

2. **Ensemble Approach:**
   - Combine Qwen3 VL (precision) + InternVL 3.5 (recall)
   - Use weighted voting based on model strengths
   - Implement confidence thresholding

3. **Fine-tune on domain-specific data:**
   - Collect larger datasets for HADR, Military Conflict, Military Showcase
   - Train specialized models for each domain
   - Target 1000+ images per domain for robust training

4. **Develop specialized AIM detection:**
   - AI-manipulated images are consistently harder to detect
   - Build models specifically for manipulation boundary detection
   - Focus on region-based analysis

5. **Add explainability:**
   - Implement attention visualization (Grad-CAM)
   - Show which regions triggered fake classification
   - Help human reviewers make informed decisions

---

## 📈 Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1 | FP | FN | Real Spec. | AIG Detect | AIM Detect |
|-------|----------|-----------|--------|----|----|----|-----------|-----------|-----------|
| **Qwen3 VL 32B** | 55.56% ⭐ | 90.00% ⭐ | 37.50% | 52.94% ⭐ | 2 ⭐ | 30 | 91.7% | 41.7% | 33.3% |
| **InternVL 3.5 8B** | 48.61% | 68.97% | 41.67% ⭐ | 51.95% | 9 | 28 ⭐ | 62.5% | 45.8% ⭐ | 37.5% ⭐ |
| **MiniCPM-V 4.5** | 41.67% | 100.00% ⭐ | 12.50% | 22.22% | 0 ⭐ | 42 | 100.0% ⭐ | 16.7% | 8.3% |
| **InternVL 2.5 8B** | 34.72% | 100.00% ⭐ | 2.08% | 4.08% | 0 ⭐ | 47 | 100.0% ⭐ | 0.0% | 4.2% |

⭐ = Best or tied for best in category

**Key Observations:**
- **Qwen3 VL** leads in overall performance and minimizes false positives
- **InternVL 3.5** catches the most fakes but with more false alarms
- **MiniCPM-V & InternVL 2.5** are too conservative - perfect precision but terrible recall
- **All models struggle with AIM** - manipulated images are consistently harder

---

## 🎬 Next Steps

1. **Review the full analysis:** See [COMPREHENSIVE_REPORT.md](COMPREHENSIVE_REPORT.md) (27 KB)
2. **Use visualizations for presentations:** 7 high-res PNG files ready for slide decks
3. **Examine domain-specific performance:** Check [7_domain_performance.png](7_domain_performance.png)
4. **Analyze raw data:** Import [detailed_results.csv](detailed_results.csv) and [domain_performance.csv](domain_performance.csv)

### Implementation Checklist:

- [ ] Deploy Qwen3 VL 32B for automated screening
- [ ] Set up InternVL 3.5 8B as secondary high-sensitivity check
- [ ] Establish human review process for all flagged content
- [ ] Implement reverse image search integration
- [ ] Test threshold optimization (try 3.5, 4.5 instead of 4.0)
- [ ] Monitor false positive and false negative rates in production
- [ ] Collect domain-specific data for fine-tuning
- [ ] Plan for model updates as AI generation tech evolves

---

## 📊 All Visualizations Included

Ready-to-use, high-resolution images for slide decks (300 DPI):

✅ **1_model_comparison.png** (489 KB) - 4-panel performance overview with all 4 models
✅ **2_confusion_matrices.png** (237 KB) - Side-by-side TP/TN/FP/FN breakdown for all models
✅ **3_dataset_composition.png** (395 KB) - 4-panel dataset statistics (categories, domains, scenarios, time)
✅ **4_performance_table.png** (180 KB) - Summary metrics as formatted table
✅ **5_detection_rates.png** (216 KB) - AIG vs AIM comparison chart for all models
✅ **6_confusion_definitions.png** (622 KB) - Explanation of metrics in context
✅ **7_domain_performance.png** (177 KB) - Performance by domain (NEW)

---

## 🔗 Quick Links

- **Full Technical Report:** [COMPREHENSIVE_REPORT.md](COMPREHENSIVE_REPORT.md) - 27 KB comprehensive analysis
- **Usage Guide:** [README.md](README.md) - How to use all files for presentations
- **Raw Data:** [detailed_results.csv](detailed_results.csv) & [domain_performance.csv](domain_performance.csv)
- **Dataset Stats:** [dataset_summary.txt](dataset_summary.txt)
- **Master Index:** [INDEX.txt](INDEX.txt) - Complete file listing

---

## ⚡ Quick Stats Reference

**Dataset:**
- 72 total images (perfectly balanced)
- 3 domains: HADR, Military Conflict, Military Showcase (24 each)
- 3 categories: AIG, AIM, REAL (24 each)
- 50/50 day/night split

**Best Model:** Qwen3 VL 32B
- 55.56% accuracy (40/72 correct)
- 90% precision (18/20 AI predictions correct)
- 37.5% recall (18/48 fakes caught)
- Only 2 false positives in entire dataset

**Key Finding:**
- AI-Manipulated images are ~8% harder to detect than AI-Generated
- Even best model misses 62.5% of fakes → human verification essential
- No single model fits all use cases → deploy based on risk tolerance

---

## 🚨 Critical Warning

**Do NOT rely solely on automated detection for high-stakes decisions.**

All models tested miss the majority of deepfakes. Human expert review is **mandatory** for:
- Military intelligence assessments
- Emergency response decisions
- News publication verification
- Legal/forensic applications
- Any scenario where errors have serious consequences

Use AI detection as a **screening tool**, not a final arbiter of authenticity.

---

**Questions or need custom analysis?**
All source code is available in `generate_report_updated.py` for modification and rerunning.

---

**Report prepared:** December 8, 2025
**Evaluation dates:** November 26 & December 2, 2024
**Total evaluations:** 1,440 (72 images × 5 runs × 4 models)
**Domains:** HADR, Military Conflict, Military Showcase
