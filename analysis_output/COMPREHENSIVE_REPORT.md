# Deepfake Detection Evaluation Report
## Vision-Language Model Performance Analysis

**Report Generated:** December 8, 2025
**Evaluation Periods:** November 26, 2024 & December 2, 2024
**Models Evaluated:** 4 Vision-Language Models
**Total Images Tested:** 72 unique images across 3 domains

---

## Executive Summary

This report presents a comprehensive evaluation of four Vision-Language Models (VLMs) for deepfake detection across three critical domains: **Humanitarian Assistance & Disaster Response (HADR)**, **Military Conflict**, and **Military Showcase** scenarios. The evaluation reveals significant performance differences across models, with **Qwen3 VL 32B** achieving the highest overall accuracy.

**Key Findings:**
- **Qwen3 VL 32B achieved the best performance** (55.56% accuracy) with excellent precision (90%) and moderate recall (37.5%)
- **InternVL 3.5 8B ranks second** (48.61% accuracy) with best recall among all models (41.67%)
- **AI-Manipulated (AIM) images are consistently harder to detect** than fully AI-Generated (AIG) images by ~8% across capable models
- **Dataset spans 3 domains equally:** HADR, Military Conflict, and Military Showcase (24 images each)
- **Even the best model misses >60% of fake images** - human verification remains critical

---

## 1. Evaluation Methodology

### 1.1 Prompts Used

#### System Prompt:
```
You are a deepfake detection assistant. You will receive an image or video and an analysis prompt.
After completing the analyses, you will give a final 0–10 score:
0 = certainly real, 10 = certainly fake. Anything above 4 is a cause for concern that its likely AI generated.
Provide clear reasoning for all steps.
```

#### Analysis Prompt:
```
Assess the media and estimate whether it is more likely to be real, manipulated, or AI-generated.
Explain your decision and rate your confidence as: High, Medium, or Low.
Factors to consider:
- Semantic coherence
- Lighting, textures, artifacts
- Any signs of digital editing
```

### 1.2 Classification Logic

- **Score > 4** → Classified as "AI Generated"
- **Score ≤ 4** → Classified as "Real"

### 1.3 Consensus Voting

Each image was evaluated **5 times** per model, and the **majority vote** determined the final consensus classification. This approach reduces the impact of random variation and provides more stable predictions.

---

## 2. Dataset Summary

### 2.1 Overall Composition

| Metric | Value |
|--------|-------|
| **Total Images** | 72 |
| **Real Images** | 24 (33.3%) |
| **AI-Generated (AIG)** | 24 (33.3%) |
| **AI-Manipulated (AIM)** | 24 (33.3%) |
| **Models Evaluated** | 4 |
| **Runs per Image** | 5 |

The dataset is **perfectly balanced** across three categories, ensuring unbiased evaluation.

### 2.2 Domain Distribution

| Domain | Images | Percentage | Description |
|--------|--------|------------|-------------|
| **HADR** | 24 | 33.3% | Humanitarian Assistance & Disaster Response (earthquakes, fires, floods, storms) |
| **Military Conflict** | 24 | 33.3% | Active combat scenarios (airstrikes, ground battles, convoys, urban damage) |
| **Military Showcase** | 24 | 33.3% | Military demonstrations (parades, marching, live fire exercises, navigation) |

Each domain contains **8 AIG + 8 AIM + 8 REAL images**, maintaining perfect balance.

### 2.3 Category Definitions

#### **REAL**
Authentic, unmanipulated photographs of real scenarios across all three domains.

#### **AIG (AI-Generated)**
Fully synthetic images created entirely by generative AI models (e.g., DALL-E, Midjourney, Stable Diffusion). These images have no basis in real photography.

#### **AIM (AI-Manipulated)**
Real photographs that have been edited or altered using AI tools (e.g., AI inpainting, deepfake face swaps, generative fill). These maintain some authentic elements but contain artificial modifications.

### 2.4 Scenario Distribution

| Scenario | Count | Domain |
|----------|-------|--------|
| **Fire** | 6 | HADR (natural fires, wildfires) |
| **Earthquake** | 6 | HADR |
| **Flood** | 6 | HADR |
| **Storm** | 6 | HADR |
| **Airstrike** | 6 | Military Conflict |
| **Ground Battle** | 6 | Military Conflict |
| **Convoy** | 6 | Military Conflict |
| **Urban Damage** | 6 | Military Conflict |
| **Parade** | 6 | Military Showcase |
| **Marching** | 6 | Military Showcase |
| **Live Fire** | 6 | Military Showcase (training exercises) |
| **Navigation** | 6 | Military Showcase |

**Note:** "Fire" (HADR) and "Live Fire" (Military Showcase) are distinct scenarios - the former refers to natural disaster fires, while the latter refers to military training exercises.

### 2.5 Time of Day Distribution

| Time | Count | Percentage |
|------|-------|------------|
| **Day** | 36 | 50.0% |
| **Night** | 36 | 50.0% |

The even split ensures models are tested under varying lighting conditions.

---

## 3. Model Performance Analysis

### 3.1 Overall Performance Summary (Ranked by Accuracy)

| Rank | Model | Accuracy | Precision | Recall | F1 Score | TP | TN | FP | FN |
|------|-------|----------|-----------|--------|----------|----|----|----|----|
| 🥇 | **Qwen3 VL 32B** | **55.56%** | **90.00%** | 37.50% | **52.94%** | 18 | 22 | 2 | 30 |
| 🥈 | **InternVL 3.5 8B** | 48.61% | 68.97% | **41.67%** | 51.95% | 20 | 15 | 9 | 28 |
| 🥉 | **MiniCPM-V 4.5** | 41.67% | 100.00% | 12.50% | 22.22% | 6 | 24 | 0 | 42 |
| 4 | **InternVL 2.5 8B** | 34.72% | 100.00% | 2.08% | 4.08% | 1 | 24 | 0 | 47 |

**Winner:** Qwen3 VL 32B (best accuracy, precision, and F1 score)

### 3.2 Confusion Matrix Definitions (Context-Specific)

In the context of deepfake detection for critical scenarios (disaster response, military intelligence):

#### **True Positive (TP)** ✅
- **Definition:** Model correctly identifies an AI-generated/manipulated image as "AI Generated"
- **Impact:** Successfully caught a fake that could spread misinformation
- **Example:** An AI-generated military convoy photo is correctly flagged as fake, preventing false intelligence reports

#### **True Negative (TN)** ✅
- **Definition:** Model correctly identifies an authentic image as "Real"
- **Impact:** Properly verified genuine footage, allowing it to be trusted
- **Example:** A real earthquake photo is correctly identified as authentic, enabling appropriate emergency response

#### **False Positive (FP)** ⚠️
- **Definition:** Model incorrectly identifies an authentic image as "AI Generated"
- **Impact:** Falsely discredited legitimate footage - damages trust & credibility
- **Example:** A real military operation photo is wrongly flagged as fake, undermining intelligence credibility
- **Consequence:** Can lead to "crying wolf" syndrome where users ignore warnings, or dismissal of genuine evidence

#### **False Negative (FN)** 🚨
- **Definition:** Model incorrectly identifies a fake image as "Real"
- **Impact:** Failed to detect misinformation - allows fakes to spread unchecked
- **Example:** An AI-generated disaster scene is accepted as real, potentially causing panic or misdirecting aid
- **Consequence:** Can cause public panic, misallocate resources, create false military intelligence, or undermine response efforts

### 3.3 Trade-off Analysis

**High Precision, Low Recall (Qwen3 VL, InternVL 2.5, MiniCPM):**
- Pros: Avoids false accusations, maintains trust in real content
- Cons: Misses many fakes, allows misinformation to spread

**Balanced Approach (InternVL 3.5):**
- Pros: Catches more fakes while maintaining reasonable precision
- Cons: Higher false positive rate may reduce user trust over time

**Optimal balance depends on use case:**
- **Emergency responders** → Prefer low FP (can't afford to ignore real disasters)
- **Military intelligence** → Prefer low FP (false alarms undermine credibility)
- **Social media moderation** → Prefer low FN (can't allow viral misinformation)
- **Research/Analysis** → Prefer balanced approach with human verification

---

## 4. AIG vs AIM Performance Breakdown

### 4.1 Detection Rate Comparison

| Model | AIG Detection | AIM Detection | Difference | Real Specificity |
|-------|---------------|---------------|------------|------------------|
| **Qwen3 VL 32B** | 41.7% (10/24) | 33.3% (8/24) | **+8.4%** | 91.7% (22/24) |
| **InternVL 3.5 8B** | 45.8% (11/24) | 37.5% (9/24) | **+8.3%** | 62.5% (15/24) |
| **MiniCPM-V 4.5** | 16.7% (4/24) | 8.3% (2/24) | **+8.4%** | 100.0% (24/24) |
| **InternVL 2.5 8B** | 0.0% (0/24) | 4.2% (1/24) | -4.2%* | 100.0% (24/24) |

*Note: InternVL 2.5's result is not meaningful due to near-zero detection overall.

### 4.2 Key Insight: AIM is Consistently Harder to Detect

**All capable models (Qwen3 VL, InternVL 3.5, MiniCPM-V) detected AIG images ~8% more easily than AIM images.**

**Why AI-Manipulated Images Are Harder:**
1. **Authentic Foundation:** AIM images start with real photos, retaining genuine photographic characteristics (natural lighting, realistic textures, correct physics)
2. **Subtle Modifications:** AI manipulations may only alter specific regions (e.g., faces, objects, backgrounds), leaving most of the image authentic
3. **Mixed Signals:** Models receive conflicting cues - some areas appear real, others show AI artifacts
4. **Advanced Techniques:** Modern AI editing tools (e.g., Adobe Firefly, Photoshop Generative Fill, DALL-E inpainting) produce seamless blends that are difficult to detect

**Why Fully AI-Generated Images Are Easier:**
1. **Synthetic Throughout:** Entire image is generated, more opportunities for detectable artifacts
2. **Consistency Issues:** AI struggles to maintain perfect consistency across all elements (shadows, reflections, perspective)
3. **Physics Violations:** Generated images may contain subtle impossibilities in lighting, shadows, or material properties
4. **Texture Patterns:** AI-generated textures often have detectable repetitive patterns or unnatural "smoothness"
5. **Complex Details:** Text, fine details, and intricate patterns often reveal generation artifacts

---

## 5. Detailed Model Analysis

### 5.1 Qwen3 VL 32B - BEST OVERALL PERFORMANCE 🏆

**Evaluation Date:** December 2, 2024
**Overall Performance:** WINNER

**Performance Characteristics:**
- **Highest Overall Accuracy:** 55.56% (40/72 correct)
- **Excellent Precision:** 90.00% (18/20 AI predictions correct)
- **Moderate Recall:** 37.50% (detects more than 1/3 of fakes)
- **Best F1 Score:** 52.94% (best overall balance)

**Strengths:**
- Best accuracy among all models tested
- Excellent precision (90%) - only 2 false positives
- Strong real image identification (91.7% specificity)
- Most reliable when it flags something as fake
- Good balance between catching fakes and avoiding false alarms

**Weaknesses:**
- Still misses 62.5% of fake images (30/48)
- Slightly lower recall than InternVL 3.5
- Still struggles with AIM detection (33.3%)

**Confusion Matrix:**
```
                 Predicted: Real    Predicted: AI
Actual: Real         22 (TN)            2 (FP)
Actual: AI           30 (FN)           18 (TP)
```

**Use Case Recommendation:**
- **✅ RECOMMENDED FOR:** General-purpose deepfake detection where balance is critical
- **✅ Military intelligence analysis** - Low FP rate maintains credibility
- **✅ News/journalism verification** - High precision ensures flagged content is likely fake
- **✅ Automated screening systems** - Best overall performance for first-pass filtering
- **⚠️ Still requires human verification** for all positive detections

---

### 5.2 InternVL 3.5 8B - BEST RECALL ⭐

**Evaluation Date:** November 26, 2024
**Overall Performance:** SECOND PLACE

**Performance Characteristics:**
- **Overall Accuracy:** 48.61% (35/72 correct)
- **Moderate Precision:** 68.97% (20/29 AI predictions correct)
- **Highest Recall:** 41.67% (catches most fakes among all models)
- **F1 Score:** 51.95% (second best balance)

**Strengths:**
- Highest recall (41.67%) - catches more fakes than any other model
- Best AIG detection rate (45.8%)
- Most aggressive at flagging suspicious content
- Good for scenarios where catching fakes is priority

**Weaknesses:**
- 9 false positives (37.5% of real images wrongly flagged)
- Lower precision (68.97%) compared to Qwen3 VL
- May cause "alert fatigue" in operational settings

**Confusion Matrix:**
```
                 Predicted: Real    Predicted: AI
Actual: Real         15 (TN)            9 (FP)
Actual: AI           28 (FN)           20 (TP)
```

**Use Case Recommendation:**
- **✅ Social media content moderation** - Higher sensitivity to catch viral fakes
- **✅ Research applications** - When false positives can be reviewed manually
- **✅ High-threat scenarios** - When missing a fake has severe consequences
- **⚠️ NOT for emergency response** - High FP rate could cause missed real emergencies

---

### 5.3 MiniCPM-V 4.5 - CONSERVATIVE APPROACH 🛡️

**Evaluation Date:** November 26, 2024
**Overall Performance:** THIRD PLACE

**Performance Characteristics:**
- **Overall Accuracy:** 41.67% (30/72 correct)
- **Perfect Precision:** 100% (all 6 positive predictions correct)
- **Low Recall:** 12.50% (misses 87.5% of fakes)
- **F1 Score:** 22.22%

**Strengths:**
- Zero false positives - never wrongly discredits real images
- Perfect real image identification (100% specificity)
- When it flags something as fake, it's virtually guaranteed to be fake
- No "crying wolf" problem

**Weaknesses:**
- Extremely low detection rate (catches only 6 out of 48 fakes)
- Essentially fails at primary purpose (detecting fakes)
- Particularly poor at AIM detection (8.3%)
- Over-conservative threshold

**Confusion Matrix:**
```
                 Predicted: Real    Predicted: AI
Actual: Real         24 (TN)            0 (FP)
Actual: AI           42 (FN)            6 (TP)
```

**Use Case Recommendation:**
- **✅ Emergency response contexts** - Zero false alarms critical
- **✅ High-stakes verification** - When false positives have serious consequences
- **✅ Second-opinion validator** - Confirms images flagged by other systems
- **✅ Trust-critical applications** - Maintaining credibility is paramount
- **❌ NOT for primary detection** - Misses too many fakes

---

### 5.4 InternVL 2.5 8B - NOT RECOMMENDED ⚠️

**Evaluation Date:** November 26, 2024
**Overall Performance:** FOURTH PLACE

**Performance Characteristics:**
- **Overall Accuracy:** 34.72% (worst) (25/72 correct)
- **Perfect Precision:** 100% (but based on only 1 positive prediction)
- **Nearly Zero Recall:** 2.08% (caught only 1 out of 48 fakes)
- **F1 Score:** 4.08% (worst)

**Strengths:**
- Zero false positives
- Perfect real image identification
- (These are hollow victories given its failure to detect fakes)

**Weaknesses:**
- Essentially non-functional as a deepfake detector
- Caught only 1 fake image out of 48 (2.08% detection rate)
- Zero AIG detection capability
- Minimal AIM detection (4.2%)
- Appears to default to classifying everything as "Real"

**Confusion Matrix:**
```
                 Predicted: Real    Predicted: AI
Actual: Real         24 (TN)            0 (FP)
Actual: AI           47 (FN)            1 (TP)
```

**Use Case Recommendation:**
- **❌ NOT RECOMMENDED** for any deepfake detection tasks
- May be suitable for other vision tasks but ineffective for this purpose
- Consider as baseline for "always predict Real" approach

---

## 6. Domain-Specific Performance

### 6.1 Performance by Domain

| Model | HADR Accuracy | Military Conflict Accuracy | Military Showcase Accuracy |
|-------|---------------|---------------------------|---------------------------|
| **Qwen3 VL 32B** | TBD | TBD | TBD |
| **InternVL 3.5 8B** | TBD | TBD | TBD |
| **MiniCPM-V 4.5** | TBD | TBD | TBD |
| **InternVL 2.5 8B** | TBD | TBD | TBD |

*Note: Detailed domain-specific breakdown available in visualization 7_domain_performance.png and domain_performance.csv*

### 6.2 Domain Characteristics

**HADR (Humanitarian Assistance & Disaster Response):**
- Natural disasters and emergency scenarios
- High emotional impact imagery
- Critical for emergency response decision-making
- False positives can delay real disaster response

**Military Conflict:**
- Active combat and war zone imagery
- High-stakes intelligence applications
- False positives can undermine intelligence credibility
- False negatives can lead to misinformation campaigns

**Military Showcase:**
- Non-combat military demonstrations
- Public relations and documentation
- Lower immediate risk but reputation-critical
- May contain propaganda or altered imagery

---

## 7. Key Insights & Recommendations

### 7.1 Primary Findings

1. **Qwen3 VL 32B is the clear winner** with 55.56% accuracy and excellent precision (90%)
2. **InternVL 3.5 8B has highest recall** (41.67%) but more false positives
3. **AI-Manipulated images are consistently ~8% harder to detect** than fully AI-generated images
4. **No model achieves >42% recall:** Even the best models miss majority of fakes
5. **Precision-Recall Trade-off is significant:** High precision models are overly conservative
6. **Dataset balance is maintained:** 33/33/33 split across categories AND domains

### 7.2 Recommendations for Immediate Deployment

#### **For High-Accuracy Applications:**
**Use: Qwen3 VL 32B**
- Best overall performance (55.56% accuracy)
- Excellent precision (90%) minimizes false accusations
- Only 2 false positives in entire dataset
- Suitable for: Military intelligence, news verification, automated screening

#### **For Maximum Fake Detection:**
**Use: InternVL 3.5 8B**
- Highest recall (41.67%)
- Catches most fakes among all models
- Accept higher false positive rate (37.5%)
- Suitable for: Social media moderation, research, high-threat scenarios

#### **For Zero False Alarms:**
**Use: MiniCPM-V 4.5**
- Perfect precision (100%)
- Zero false positives
- Very low recall (12.5%)
- Suitable for: Emergency response, trust-critical applications, second-opinion validation

#### **For Any Serious Application:**
**Use: Multi-Stage Verification Pipeline**
```
Stage 1: Automated Detection (Qwen3 VL or InternVL 3.5)
    ↓
Stage 2: Reverse Image Search + Metadata Analysis
    ↓
Stage 3: Human Expert Review (for flagged content)
    ↓
Stage 4: Contextual Verification (source credibility, timing, consistency)
```

### 7.3 Recommendations for Future Work

#### **Short-term Improvements:**

1. **Threshold Optimization:**
   - Current threshold (score > 4) may be suboptimal
   - Test thresholds from 3.0 to 5.0 in 0.5 increments
   - Create ROC curves to find optimal operating point
   - Consider domain-specific thresholds

2. **Increase Consensus Runs:**
   - Move from 5 to 7 or 9 runs per image
   - Analyze variance in predictions
   - Identify images with high disagreement

3. **Ensemble Approach:**
   - Combine Qwen3 VL (precision) + InternVL 3.5 (recall)
   - Use weighted voting based on model strengths
   - Implement confidence thresholding

4. **Prompt Engineering:**
   ```
   Enhanced Analysis Prompt:

   Analyze this image for signs of AI generation or manipulation.

   CRITICAL FOCUS AREAS:
   1. Lighting Consistency: Are shadows and highlights natural across all regions?
   2. Boundary Artifacts: Look for blending issues, especially in AI-manipulated areas
   3. Texture Analysis: Do materials have natural irregularities or AI smoothness?
   4. Physical Plausibility: Are reflections, perspectives, and physics correct?
   5. Detail Coherence: Do small details (text, faces, hands) appear natural?

   Pay special attention to AI-MANIPULATED images that blend real and fake elements.
   Score 0-10 where >4 indicates suspicion of AI involvement.
   ```

#### **Medium-term Enhancements:**

1. **Fine-tuning on Domain-Specific Data:**
   - Collect larger dataset for each domain (HADR, Military Conflict, Military Showcase)
   - Fine-tune models on 1000+ images per domain
   - Develop domain-specific detection models

2. **Specialized AIM Detection:**
   - Train separate model specifically for manipulation detection
   - Focus on boundary artifact detection
   - Implement region-based analysis

3. **Explainability Integration:**
   - Add attention visualization (Grad-CAM, SHAP)
   - Identify which regions triggered fake classification
   - Provide explanations to human reviewers

4. **Metadata Analysis:**
   - Extract EXIF data when available
   - Analyze compression artifacts
   - Check for AI generation signatures

#### **Long-term Research Directions:**

1. **3-Way Classification:**
   - Modify system to explicitly distinguish REAL/AIG/AIM
   - Train models to identify manipulation type
   - Develop separate metrics for each class

2. **Adversarial Training:**
   - Train on latest AI generation techniques
   - Include images from newest models (DALL-E 3, Midjourney V6, etc.)
   - Continuous model updates as generation tech evolves

3. **Cross-Domain Transfer Learning:**
   - Evaluate if HADR-trained models work on Military domains
   - Study domain-specific vs. general detection approaches

4. **Real-Time Detection:**
   - Optimize models for speed
   - Deploy edge-compatible versions
   - Enable field use for journalists and first responders

5. **Temporal Analysis:**
   - For video deepfakes, analyze frame-to-frame consistency
   - Detect temporal artifacts in generated sequences

---

## 8. Limitations & Future Considerations

### 8.1 Current Limitations

1. **Limited Dataset Size:** 72 images is small for robust evaluation
2. **Single-Frame Analysis:** Videos not thoroughly tested
3. **Static Prompts:** No adaptive questioning based on image type
4. **No Uncertainty Quantification:** Models don't express confidence clearly
5. **Evolving AI Generation:** Results may not generalize to newest AI models

### 8.2 Generalization Concerns

- **Newer Generation Models:** Evaluation uses specific AI generation tools; newer models may be harder to detect
- **Cross-Domain Performance:** Models may perform differently on other domains (political, entertainment, etc.)
- **Cultural Context:** Dataset may not represent global diversity of imagery
- **Adversarial Attacks:** Sophisticated actors may craft images specifically to evade detection

### 8.3 Ethical Considerations

- **False Accusations:** False positives can harm reputations, especially for citizen journalists
- **Censorship Risk:** Over-aggressive detection may suppress legitimate content
- **Automation Bias:** Users may over-trust automated systems
- **Access Inequality:** Advanced detection tools may only be available to well-resourced organizations

---

## 9. Conclusion

This evaluation demonstrates that **deepfake detection in critical scenarios (disaster response, military intelligence) remains a significant challenge** for current Vision-Language Models. While **Qwen3 VL 32B** shows promising performance with 55.56% accuracy and excellent precision, even the best model misses more than 60% of fake images.

### **Critical Takeaways:**

1. **Qwen3 VL 32B is the best current option** for balanced, high-accuracy detection
2. **AI-manipulated images (real photos altered with AI) are significantly harder to detect** than fully AI-generated images, presenting a growing challenge as manipulation tools become more sophisticated
3. **No automated system is sufficient alone** - human verification is essential for high-stakes decisions
4. **Multi-stage verification pipelines** combining AI detection, reverse image search, metadata analysis, and human review provide the best results
5. **Continuous updates are critical** as AI generation techniques evolve rapidly

### **Path Forward:**

Effective deepfake detection for humanitarian and military contexts will require:
- **Ensemble approaches** combining multiple specialized models
- **Domain-specific fine-tuning** on disaster and military imagery
- **Multi-stage verification pipelines** with human-in-the-loop validation
- **Continuous model updates** as AI generation techniques evolve
- **Threshold optimization** for specific use cases and risk tolerance
- **Transparency and explainability** to build trust in detection systems

The perfect balance between catching fakes (recall) and avoiding false alarms (precision) depends heavily on the specific application context, acceptable risk levels, and consequences of errors. No single model or approach fits all use cases.

**Final Recommendation:** Deploy Qwen3 VL 32B as the primary automated screening tool, with InternVL 3.5 8B as a secondary high-sensitivity check, and mandatory human expert review for all flagged content in critical applications.

---

## Appendix A: Evaluation Metrics Formulas

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)
*"Overall correctness rate"*

**Precision** = TP / (TP + FP)
*"Of all images flagged as fake, what % were actually fake?"*

**Recall (Sensitivity)** = TP / (TP + FN)
*"Of all actual fake images, what % did we catch?"*

**Specificity** = TN / (TN + FP)
*"Of all real images, what % did we correctly identify as real?"*

**F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
*"Harmonic mean balancing precision and recall"*

---

## Appendix B: File Naming Convention

Dataset images follow this pattern:

`[DOMAIN]_[CATEGORY]_[TIME]_[SCENARIO]_[##].jpg`

**Examples:**
- `HADR_REAL_DAY_EARTHQUAKE_01.jpg` - Real daytime earthquake photo
- `HADR_AIG_NIGHT_FIRE_01.jpg` - AI-generated nighttime fire scene
- `HADR_AIM_DAY_FLOOD_01.jpg` - AI-manipulated daytime flood image
- `MilitaryConflict_AIG_NIGHT_AIRSTRIKE_01.jpg` - AI-generated nighttime airstrike
- `MilitaryShowcase_AIM_DAY_PARADE_01.jpg` - AI-manipulated parade image

Where:
- **DOMAIN:** HADR, MilitaryConflict, MilitaryShowcase
- **CATEGORY:** REAL, AIG (AI-Generated), AIM (AI-Manipulated)
- **TIME:** DAY, NIGHT
- **SCENARIO:** EARTHQUAKE, FIRE, FLOOD, STORM, AIRSTRIKE, CONVOY, GROUNDBATTLE, URBANDAMAGE, PARADE, MARCHING, LIVEFIRE, NAVIGATION

---

## Appendix C: Visualization Index

The following visualizations are included with this report:

1. **1_model_comparison.png** - Three-panel comparison showing all 4 models:
   - Overall performance metrics (accuracy, precision, recall, F1)
   - AIG vs AIM detection rates comparison
   - Category performance breakdown (AIG/AIM/Real detection)

2. **2_confusion_matrices.png** - Side-by-side confusion matrices for all 4 models with count and percentage

3. **3_dataset_composition.png** - Four-panel dataset analysis:
   - Category distribution pie chart (AIG/AIM/REAL)
   - Domain distribution pie chart (HADR/Military Conflict/Military Showcase)
   - Scenario distribution bar chart (12 scenario types)
   - Day vs night distribution

4. **4_performance_table.png** - Comprehensive summary table with all metrics for all 4 models

5. **5_detection_rates.png** - Large bar chart comparing AIG/AIM/Real detection rates across all 4 models

6. **6_confusion_definitions.png** - Visual explanation of TP/TN/FP/FN in context

7. **7_domain_performance.png** - Performance breakdown by domain (HADR, Military Conflict, Military Showcase)

---

## Appendix D: Models Tested

1. **InternVL 2.5 8B** - opengvlab/InternVL2_5-8B
2. **InternVL 3.5 8B** - opengvlab/InternVL3_5-8B
3. **MiniCPM-V 4.5** - openbmb/MiniCPM-V-4_5
4. **Qwen3 VL 32B** - Qwen3-VL-32B-Instruct

All models accessed via vLLM serving infrastructure with OpenAI-compatible API.

---

**Report End**

*For questions or additional analysis, refer to the analysis_output/ directory for all visualizations and raw data.*
