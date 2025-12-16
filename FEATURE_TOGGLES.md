# Feature Toggles: Forensic Report & Watermark Mode

**Date:** December 15, 2025
**Feature:** Added UI toggles for forensic artifacts and watermark handling

---

## Overview

Added two new toggles to the Streamlit UI to give users control over VLM analysis behavior:

1. **Send Forensic Report to VLM** - Controls whether forensic artifacts (ELA/FFT images + text) are sent to the VLM
2. **Watermark Handling Mode** - Controls how watermarks are interpreted (ignore as news logos vs analyze as AI signatures)

---

## 1. Send Forensic Report Toggle

### UI Location
- **Tab:** Detection tab
- **Section:** Advanced Settings (expandable)
- **Control:** Checkbox
- **Label:** "📊 Send Forensic Report to VLM"
- **Default:** `False` (disabled)

### Purpose
Based on user feedback: *"at this point, the forensics report is providing more noise than signal"*

This toggle allows users to disable sending forensic artifacts to the VLM, which:
- Reduces input token count by ~2000-2500 tokens (2 images + forensic text)
- Reduces Stage 2 latency (potentially 20-30% faster)
- Focuses VLM analysis on visual/physical tells only
- Useful when forensic metrics are ambiguous or misleading

### Behavior

#### When Enabled (`send_forensics=True`)
VLM receives:
- Original image
- ELA (Error Level Analysis) map image
- FFT (Frequency Spectrum) image
- Forensic text report (ELA variance, FFT pattern type, metadata)
- Forensic interpretation guide (how to read ELA/FFT)
- 4-step analysis instructions:
  1. Physical/Visual Analysis
  2. Forensic Correlation (ELA + FFT)
  3. Metadata Check
  4. Watermark Analysis

**Token Count:** ~2900 tokens (3 images + ~600 text tokens)

#### When Disabled (`send_forensics=False`)
VLM receives:
- Original image only
- Simplified 3-step analysis instructions:
  1. Physical/Visual Analysis (anatomy, physics, composition, textures)
  2. Scene Coherence (perspective, scale, artifacts)
  3. Watermark Analysis

**Token Count:** ~900 tokens (1 image + ~150 text tokens)

**Reduction:** **~2000 tokens saved** (68% reduction)

### Implementation

**Files Modified:**
- [app.py](app.py#L91-L95) - Added checkbox in Advanced Settings
- [app.py](app.py#L293) - Pass `send_forensics` parameter to `detector.detect()`
- [detector.py](detector.py#L79) - Added `send_forensics` parameter to `detect()` method
- [detector.py](detector.py#L319) - Added `send_forensics` parameter to `_two_stage_classification()`
- [detector.py](detector.py#L338-L411) - Conditional content building based on `send_forensics` flag

**Code Example:**
```python
# app.py - UI Control
send_forensics = st.checkbox(
    "📊 Send Forensic Report to VLM",
    value=False,
    help="Include ELA/FFT forensic text report in VLM analysis. Disable if forensics provide more noise than signal."
)

# detector.py - Conditional Message Building
if send_forensics:
    # Send: Forensic Report + Original + ELA + FFT + Detailed Instructions
    user_content.extend([
        {"type": "text", "text": "--- FORENSIC LAB REPORT ---"},
        {"type": "text", "text": forensic_report},
        {"type": "text", "text": "--- ORIGINAL IMAGE ---"},
        {"type": "image_url", "image_url": {"url": original_uri}},
        {"type": "text", "text": "--- ELA MAP ---"},
        {"type": "image_url", "image_url": {"url": ela_uri}},
        {"type": "text", "text": "--- FFT SPECTRUM ---"},
        {"type": "image_url", "image_url": {"url": fft_uri}},
        # ... forensic interpretation guide ...
    ])
else:
    # Send: Original image only + Simplified Instructions
    user_content.extend([
        {"type": "text", "text": "--- IMAGE TO ANALYZE ---"},
        {"type": "image_url", "image_url": {"url": original_uri}},
        # ... simplified analysis instructions ...
    ])
```

### Expected Performance Impact

**With Forensics Disabled:**
- **Token Reduction:** 2900 → 900 tokens (68% reduction)
- **Latency Improvement:** Estimated 20-30% faster Stage 2
  - Current: 119 seconds with forensics
  - Expected: 80-95 seconds without forensics
- **Still above target:** Need further optimization to reach <30s goal

**Trade-off:**
- ✅ Faster inference
- ✅ Lower cost (fewer input tokens)
- ✅ Cleaner analysis (no forensic noise)
- ⚠️ Loss of forensic correlation capability
- ⚠️ VLM cannot cross-check visual tells with ELA/FFT anomalies

---

## 2. Watermark Handling Mode

### UI Location
- **Tab:** Detection tab
- **Section:** Advanced Settings (expandable)
- **Control:** Selectbox (dropdown)
- **Label:** "🏷️ Watermark Handling"
- **Options:**
  - `ignore` - "Ignore (Treat as news logos)" **(default)**
  - `analyze` - "Analyze (Flag AI watermarks)"

### Purpose
Watermarks can be ambiguous in OSINT imagery:
- News agency logos (CNN, BBC, Reuters) are legitimate
- AI tool watermarks (Sora, NanoBanana, colored strips) indicate synthetic content
- Without guidance, VLM may incorrectly flag news logos as AI indicators

### Behavior

#### Mode: `ignore` (Default)
**Instruction sent to VLM:**
> "Ignore all watermarks, text overlays, or corner logos. Treat them as potential OSINT source attributions (e.g., news agency logos) and NOT as evidence of AI generation."

**Use Case:**
- OSINT/military imagery often has news agency watermarks
- Prevents false positives from legitimate source attributions
- Default mode for production OSINT analysis

#### Mode: `analyze`
**Instruction sent to VLM:**
> "Actively scan for known AI watermarks (e.g., 'Sora', 'NanoBanana', colored strips, AI tool logos). If found, flag as 'Suspected AI Watermark'. CAUTION: Distinguish these from standard news/TV watermarks (CNN, BBC, Reuters)."

**Use Case:**
- General deepfake detection (non-OSINT)
- When AI watermarks are expected to be present
- Research/academic analysis

### Implementation

**Files Modified:**
- [app.py](app.py#L98-L107) - Added selectbox in Advanced Settings
- [app.py](app.py#L286) - Pass `watermark_mode` parameter to `OSINTDetector` constructor
- [detector.py](detector.py#L52) - Already implemented in constructor
- [detector.py](detector.py#L562-L575) - `_get_watermark_instruction()` method (already exists)

**Code Example:**
```python
# app.py - UI Control
watermark_mode = st.selectbox(
    "🏷️ Watermark Handling",
    options=["ignore", "analyze"],
    format_func=lambda x: {
        "ignore": "Ignore (Treat as news logos)",
        "analyze": "Analyze (Flag AI watermarks)"
    }[x],
    index=0,
    help="'Ignore' treats watermarks as news agency logos. 'Analyze' actively scans for AI tool watermarks."
)

# detector.py - Constructor
def __init__(
    self,
    base_url: str,
    model_name: str,
    api_key: str = "dummy",
    context: str = "auto",
    watermark_mode: str = "ignore"  # NEW: Exposed to UI
):
    self.watermark_mode = watermark_mode
```

---

## UI Screenshot (Conceptual)

```
┌─────────────────────────────────────────────────────────────┐
│ 🕵️‍♂️ Deepfake Detection Chat                                  │
├─────────────────────────────────────────────────────────────┤
│ Select detection model: [Qwen/Qwen2.5-VL-32B ▼]            │
│                                                             │
│ OSINT Context: [Auto-Detect ▼]                             │
│                                                             │
│ ☑️ 🔍 Enable Debug Mode                                      │
│                                                             │
│ ▼ ⚙️ Advanced Settings                                      │
│   ┌───────────────────────────────────────────────────┐   │
│   │ ☐ 📊 Send Forensic Report to VLM                  │   │
│   │                                                   │   │
│   │ 🏷️ Watermark Handling: [Ignore (Treat as news...) ▼] │   │
│   └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Testing Plan

### Test Case 1: Forensics Disabled Performance
**Objective:** Measure latency improvement with forensics disabled

**Steps:**
1. Upload test image
2. Run with `send_forensics=True` (enabled)
3. Note Stage 2 latency from debug output
4. Run same image with `send_forensics=False` (disabled)
5. Compare Stage 2 latency

**Expected Results:**
- Forensics enabled: ~119 seconds (baseline)
- Forensics disabled: ~80-95 seconds (20-30% faster)
- Token reduction: 2900 → 900 tokens

### Test Case 2: Forensics Disabled Accuracy
**Objective:** Ensure physical analysis quality is maintained without forensics

**Steps:**
1. Use test set with known physical tells (extra fingers, impossible shadows, etc.)
2. Run batch evaluation with `send_forensics=False`
3. Compare accuracy/precision/recall vs baseline (with forensics)

**Expected Results:**
- Accuracy should remain similar or improve (no forensic noise)
- VLM focuses on visual tells only
- May miss some manipulation cases that only show up in ELA/FFT

### Test Case 3: Watermark Mode Toggle
**Objective:** Verify watermark instructions are correctly applied

**Steps:**
1. Upload image with AI watermark (e.g., "Sora" text)
2. Run with `watermark_mode="ignore"`
3. Check reasoning - should NOT flag watermark
4. Run with `watermark_mode="analyze"`
5. Check reasoning - SHOULD flag watermark

**Expected Results:**
- Ignore mode: "Watermark ignored as potential news source attribution"
- Analyze mode: "Suspected AI watermark detected: 'Sora'"

---

## Usage Recommendations

### When to Enable Forensics (`send_forensics=True`)
✅ High-quality images with minimal compression
✅ When ELA/FFT show clear anomalies (grid stars, local inconsistencies)
✅ Suspected manipulation cases (splicing, cloning)
✅ When latency is not a concern (<120s acceptable)

### When to Disable Forensics (`send_forensics=False`)
✅ Social media images (WhatsApp/Facebook re-compression)
✅ When forensic metrics are ambiguous or contradictory
✅ When latency is critical (<60s required)
✅ When VLM expertise in visual tells is sufficient
✅ **Current recommendation:** Disable by default until Stage 2 latency is optimized

### Watermark Mode Selection
- **OSINT/Military Context:** Use `ignore` mode (default)
- **General Deepfake Detection:** Use `analyze` mode
- **Research/Academic:** Use `analyze` mode for comprehensive watermark tracking

---

## Related Files

- [app.py](app.py) - Streamlit UI with toggle controls
- [detector.py](detector.py) - OSINT detector with conditional forensic sending
- [TIMEOUT_OPTIMIZATION.md](TIMEOUT_OPTIMIZATION.md) - Performance optimization tracking
- [UNIFIED_DESIGN_SPEC.md](UNIFIED_DESIGN_SPEC.md) - System architecture specification

---

## Commit Summary

**Changes:**
- Added "Send Forensic Report to VLM" checkbox (default: disabled)
- Added "Watermark Handling" selectbox (default: ignore)
- Implemented conditional message building in `detector.py`
- Reduced input tokens from ~2900 to ~900 when forensics disabled (68% reduction)

**Impact:**
- User control over VLM analysis behavior
- Potential 20-30% latency improvement when forensics disabled
- Better handling of OSINT watermark ambiguity

**Status:** ✅ Implementation complete, ready for testing
