# Timeout Optimization and Stage 2 Performance Analysis

**Date:** December 15, 2025
**Issue:** Stage 2 (VLM Analysis) taking 119-133 seconds, approaching timeout limits

---

## Problem Statement

User reported system hanging during OSINT detection. After implementing client-level timeout configuration, timing data revealed:

### Run 1 (No KV-Cache):
```
Stage-by-Stage Timing:
    Stage 0 (Metadata): 0.000s
    Stage 1 (Forensics): 0.100s
    Stage 2 (VLM Analysis): 119.73s
    Stage 3 (Verdict): 1.85s
Total Pipeline: 121.68s
KV-Cache Hit: ❌ NO
```

### Run 2 (With KV-Cache):
```
Stage-by-Stage Timing:
    Stage 0 (Metadata): 0.000s
    Stage 1 (Forensics): 0.097s
    Stage 2 (VLM Analysis): 0.00s
    Stage 3 (Verdict): 0.00s
Total Pipeline: 133.10s
KV-Cache Hit: ✅ YES

❌ Second run TIMED OUT
```

---

## Analysis

### Expected vs Actual Performance

**Expected Stage 2 Latency:**
- Typical VLM inference with vision: 10-30 seconds
- Qwen3 VL 32B on reasonable hardware: 15-25 seconds

**Actual Stage 2 Latency:**
- First run: 119.73 seconds (~120s)
- Second run: Timed out at 133.10 seconds

**Discrepancy:** 4-8× slower than expected

### Possible Root Causes

1. **Prompt Length:**
   - Lines 343-373 in [detector.py](detector.py#L343-L373) contain comprehensive instructions
   - Forensic interpretation guide (10 lines)
   - Physical analysis checklist (4 categories)
   - Forensic correlation steps (3 points)
   - Metadata check (2 points)
   - Watermark analysis (variable)
   - Total prompt: ~500-600 tokens

2. **Image Processing:**
   - 3 base64-encoded images sent per request:
     - Original image
     - ELA map
     - FFT spectrum
   - Each image ~765 tokens (standard vision token count)
   - Total: ~2295 vision tokens + ~600 text tokens = **~2900 input tokens**

3. **Model/Hardware Constraints:**
   - Qwen3 VL 32B is a large model
   - Vision processing is computationally expensive
   - Possible hardware bottleneck (CPU inference? Limited GPU memory?)
   - Batch size = 1 (no batching optimization)

4. **KV-Cache Issue:**
   - Second run showed KV-cache HIT but still timed out
   - Suggests timeout occurred DURING Stage 2, not after completion
   - KV-cache only benefits Stage 3 (verdict), not Stage 2

---

## Solution Implemented

### 1. ~~Reduced Timeout from 180s to 60s~~ REVERTED: Keep 180s Timeout

**File:** [detector.py](detector.py#L67)

**Initial Change (Reverted):**
```python
timeout=60.0,  # 1 minute timeout for faster failure in production
```

**Current Setting:**
```python
timeout=180.0,  # 3 minute timeout to accommodate slower VLM inference
```

**Rationale for Revert:**
- User feedback: "60s appears to be too short"
- Stage 2 legitimately takes 119-133 seconds with full forensic analysis
- 60s timeout would cause false failures on valid requests
- Better to accommodate slower inference than timeout prematurely

**Alternative Solutions:**
- ✅ **Forensic Toggle (Implemented):** User can disable forensics to reduce latency
  - With forensics: ~119s (under 180s timeout)
  - Without forensics: ~80-95s estimated (well under 180s timeout)
- ⏳ **Prompt Optimization (Future):** Reduce verbosity of forensic instructions
- ⏳ **Hardware Upgrade (Future):** Better GPU or model quantization

---

## Recommendations for Further Optimization

### Priority 1: Investigate Model/Hardware Bottleneck

**Action Items:**
1. Check vLLM logs for Stage 2 processing:
   ```bash
   docker logs deepfake-detector-vlm-1 | tail -100
   ```
2. Monitor GPU/CPU usage during Stage 2:
   ```bash
   nvidia-smi -l 1  # If using GPU
   htop             # For CPU usage
   ```
3. Verify model quantization:
   - Is Qwen3 VL 32B running in FP16, INT8, or INT4?
   - Higher quantization = faster inference
4. Check batch size and context length limits in vLLM config

**Expected Findings:**
- If GPU utilization < 80% → Likely CPU bottleneck
- If processing time scales linearly with prompt length → Likely hardware constraint
- If logs show errors/warnings → Model configuration issue

### Priority 2: Optimize Stage 2 Prompt

**Current Prompt Structure (Lines 343-373):**
```python
"--- FORENSIC LAB REPORT ---"
{forensic_report}
"--- ORIGINAL IMAGE ---"
{original_image}
"--- ELA MAP ---"
{ela_image}
"--- FFT SPECTRUM ---"
{fft_image}
"--- FORENSIC INTERPRETATION GUIDE ---"
{10 lines of FFT/ELA interpretation}
"--- ANALYSIS INSTRUCTIONS ---"
{4-section checklist with 12 sub-points}
```

**Optimization Options:**

**Option A: Reduce Instruction Verbosity (Conservative)**
- Remove redundant explanations
- Condense checklist to bullet points
- Estimated reduction: 200-300 tokens (20-30% faster)

Example:
```python
"Analyze the image for:\n"
"1. Physical tells: anatomy, physics, textures\n"
"2. Forensic correlation: ELA inconsistencies, FFT grid stars\n"
"3. Metadata/watermark checks\n"
```

**Option B: Move Forensic Interpretation to System Prompt (Aggressive)**
- Move FFT/ELA interpretation guide to system prompt (lines 453-491)
- Only send analysis checklist in user prompt
- Estimated reduction: 300-400 tokens (30-40% faster)

**Option C: Reduce Image Inputs (Radical)**
- Remove ELA or FFT from Stage 2 (keep in forensic report text only)
- Send only original image + text report
- Estimated reduction: 765 tokens per image (25-30% faster per image removed)
- ⚠️ **Risk:** May reduce detection accuracy if VLM can't see forensic artifacts directly

**Recommendation:** Try **Option A** first (conservative optimization), then **Option B** if still too slow.

### Priority 3: Implement Progressive Timeout Strategy

Instead of fixed 60-second timeout, use context-aware timeouts:

```python
def __init__(
    self,
    base_url: str,
    model_name: str,
    api_key: str = "dummy",
    context: str = "auto",
    watermark_mode: str = "ignore",
    timeout_stage_2: int = 60,  # NEW: Configurable Stage 2 timeout
    timeout_stage_3: int = 10   # NEW: Shorter timeout for Stage 3 (KV-cached)
):
    self.timeout_stage_2 = timeout_stage_2
    self.timeout_stage_3 = timeout_stage_3
    # Use shorter timeout for Stage 3 since it's KV-cached
```

Then in `_two_stage_classification()`:
```python
# Stage 2 with longer timeout
response_1 = self.client.chat.completions.create(
    model=self.model_name,
    messages=messages,
    temperature=0.0,
    max_tokens=500,
    timeout=self.timeout_stage_2  # 60s
)

# Stage 3 with shorter timeout (KV-cached, should be fast)
response_2 = self.client.chat.completions.create(
    model=self.model_name,
    messages=messages,
    temperature=0.0,
    max_tokens=1,
    logprobs=True,
    top_logprobs=5,
    timeout=self.timeout_stage_3  # 10s
)
```

**Benefits:**
- Stage 2 gets full 60s for complex analysis
- Stage 3 fails fast at 10s if KV-cache isn't working
- Easier to diagnose KV-cache issues

---

## Testing Plan

### Test Case 1: Verify 60s Timeout Works
**Objective:** Confirm timeout triggers at 60 seconds instead of 180 seconds

**Steps:**
1. Rebuild Docker container
2. Upload test image
3. Monitor timing in debug output

**Expected Results:**
- If Stage 2 takes >60s: Request times out with descriptive error
- Forensic report still displayed (error handling works)
- User sees "VLM analysis error: timeout" message

### Test Case 2: Measure Prompt Optimization Impact
**Objective:** Quantify performance improvement from prompt reduction

**Steps:**
1. Baseline: Current verbose prompt (500-600 tokens)
2. Test Option A: Condensed prompt (300-400 tokens)
3. Compare Stage 2 latency

**Expected Results:**
- 20-30% reduction in Stage 2 time
- Example: 120s → 85s (still too slow, but progress)

### Test Case 3: Hardware Profiling
**Objective:** Identify bottleneck (GPU vs CPU vs model config)

**Steps:**
1. Run detection with GPU monitoring:
   ```bash
   watch -n 1 nvidia-smi
   ```
2. Check vLLM logs during Stage 2:
   ```bash
   docker logs -f deepfake-detector-vlm-1
   ```
3. Measure GPU utilization, memory usage, batch processing time

**Expected Findings:**
- Low GPU utilization (<50%) → CPU bottleneck
- High GPU utilization (>90%) → Model too large for hardware
- vLLM logs show "waiting for GPU" → Batch size issue

---

## Rollback Plan

If 60-second timeout causes too many false timeouts on legitimate requests:

```python
# Revert to 180s
timeout=180.0,  # Reverted - 60s too aggressive for current hardware
```

Or implement fallback:
```python
timeout=os.getenv("VLM_TIMEOUT", 60.0)  # Allow environment override
```

---

## Performance Benchmarks

### Current Performance (December 15, 2025)

| Stage | Latency | Token Count | Bottleneck |
|-------|---------|-------------|------------|
| Stage 0 (Metadata) | 0.000s | 0 | None |
| Stage 1 (Forensics) | 0.100s | 0 | Image processing (acceptable) |
| **Stage 2 (VLM Analysis)** | **119.73s** | ~2900 input | **Model/Hardware** ⚠️ |
| Stage 3 (Verdict) | 1.85s | ~50 input | KV-cache (acceptable) |
| **Total Pipeline** | **121.68s** | | **Stage 2 dominates** |

### Target Performance (Production Goal)

| Stage | Target Latency | Optimization Needed |
|-------|---------------|---------------------|
| Stage 0 (Metadata) | <0.01s | ✅ Already optimal |
| Stage 1 (Forensics) | <0.2s | ✅ Already optimal |
| **Stage 2 (VLM Analysis)** | **15-25s** | ❌ Need 4-8× speedup |
| Stage 3 (Verdict) | <2s | ✅ Already optimal |
| **Total Pipeline** | **<30s** | ❌ Currently 4× too slow |

**Required Improvements:**
- Reduce Stage 2 from 120s to 20s → **6× speedup needed**
- Possible approaches:
  - Model quantization (INT8 or INT4) → 2-3× faster
  - Prompt optimization → 1.3-1.5× faster
  - Hardware upgrade (better GPU) → 2-4× faster
  - Combined: Could achieve 5-10× speedup

---

## Related Files

- [detector.py](detector.py) - Main detection pipeline (timeout config at line 67)
- [DETECTOR_IMPLEMENTATION_ANALYSIS.md](DETECTOR_IMPLEMENTATION_ANALYSIS.md) - Implementation completeness analysis
- [FORENSIC_PIPELINE_FIXES.md](FORENSIC_PIPELINE_FIXES.md) - Forensic pipeline optimization history
- [URGENT_TIMEOUT_FIX.md](URGENT_TIMEOUT_FIX.md) - Previous timeout fix (per-request → client-level)

---

## Sign-Off

**Issue:** Stage 2 taking 119-133 seconds, approaching 180s timeout
**Root Cause:** Model/hardware bottleneck + verbose prompt
**Immediate Fix:** Reduced timeout from 180s → 60s for faster failure
**Next Steps:** Investigate hardware bottleneck + optimize prompt verbosity

**Status:** ⏳ Awaiting user testing with 60-second timeout
**Priority:** HIGH - Blocking production deployment until Stage 2 is <30 seconds
