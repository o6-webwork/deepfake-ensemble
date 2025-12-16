# App.py Tab 1 Update Guide for OSINT Detection

This guide outlines the changes needed to integrate the new OSINTDetector into Tab 1 of app.py.

## 1. Add Imports

At the top of app.py (after line 20), add:

```python
from detector import OSINTDetector
```

## 2. Add Session State Variables

After line 34, add:

```python
if "osint_context" not in st.session_state:
    st.session_state.osint_context = "auto"  # auto/military/disaster/propaganda
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "osint_result" not in st.session_state:
    st.session_state.osint_result = None  # Stores OSINTDetector result
```

## 3. Add OSINT Controls (After Model Selector)

After line 55 (detect_model_key assignment), add:

```python
# OSINT Context Selector
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

# Debug Mode Toggle
debug_mode = st.checkbox(
    "🔍 Enable Debug Mode",
    value=st.session_state.debug_mode,
    help="Show detailed forensic reports, VLM reasoning, and raw logprobs"
)
st.session_state.debug_mode = debug_mode
```

## 4. Replace Forensic Analysis (Lines 201-250)

Replace the current forensic analysis block (starts at line 201 `if new_upload and analysis_image:`) with:

```python
if new_upload and analysis_image:
    with st.spinner("🔬 Running OSINT detection pipeline..."):
        try:
            # Convert PIL Image to bytes
            img_bytes = io.BytesIO()
            analysis_image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()

            # Create OSINT detector
            config = MODEL_CONFIGS[detect_model_key]
            detector = OSINTDetector(
                base_url=config["base_url"],
                model_name=config["model_name"],
                api_key=config.get("api_key", "dummy"),
                context=st.session_state.osint_context
            )

            # Run detection with debug mode
            result = detector.detect(
                img_bytes,
                debug=st.session_state.debug_mode
            )

            # Store result and extract forensic artifacts for display
            st.session_state.osint_result = result

            # Generate artifacts for display (if not already in result)
            # Note: detector already generated these internally, but we need bytes for display
            from forensics import ArtifactGenerator
            ag = ArtifactGenerator()
            ela_bytes = ag.generate_ela(img_bytes)
            fft_bytes, _ = ag.generate_fft_preprocessed(img_bytes)
            st.session_state.forensic_artifacts = (ela_bytes, fft_bytes)

            # Create assistant message
            tier = result['tier']
            confidence_pct = result['confidence'] * 100
            reasoning = result['reasoning']

            # Determine color coding
            if tier == "Deepfake":
                tier_emoji = "🚨"
                tier_color = "red"
            elif tier == "Suspicious":
                tier_emoji = "⚠️"
                tier_color = "orange"
            else:
                tier_emoji = "✅"
                tier_color = "green"

            assistant_msg = f"""**Model:** {detect_model_display}
**OSINT Context:** {osint_context.capitalize()}

**{tier_emoji} Classification: {tier}**
**Confidence:** {confidence_pct:.1f}%

**VLM Reasoning:**
{reasoning}

**Forensic Report:**
```
{result['forensic_report']}
```
"""

            # Add debug information if enabled
            if st.session_state.debug_mode and 'debug' in result:
                debug = result['debug']

                assistant_msg += f"""

---
### 🔬 Debug: Forensic Lab Report (Raw Data)

**EXIF Metadata:**
```
{chr(10).join([f"{k}: {v}" for k, v in debug['exif_data'].items()]) if debug['exif_data'] else '(No EXIF data found)'}
```

**ELA Analysis:**
- Variance Score: {debug['ela_variance']:.2f}
- Threshold: <2.0 (AI indicator)

**FFT Analysis:**
- Pattern Type: {debug['fft_pattern']}
- Peaks Detected: {debug['fft_peaks']}

**OSINT Context Applied:** {debug['context_applied'].capitalize()}

---
### 🧠 VLM Analysis Output

**Full Reasoning:**
{reasoning}

**API Metadata:**
- Model: {detect_model_display}
- Request 1 Latency: {debug['request_1_latency']:.2f}s
- Request 2 Latency: {debug['request_2_latency']:.2f}s (⚡ {((1 - debug['request_2_latency']/debug['request_1_latency']) * 100):.1f}% faster via KV-cache)
- Request 1 Tokens: ~{debug['request_1_tokens']}
- Request 2 Tokens: ~{debug['request_2_tokens']}

---
### 📊 Logprobs & Verdict Extraction

**Top K=5 Tokens:**
"""
                # Format top-k logprobs as table
                for i, (token, logprob) in enumerate(debug['top_k_logprobs'][:5], 1):
                    prob = math.exp(logprob)
                    interpretation = ""
                    if token in detector.REAL_TOKENS:
                        interpretation = "(REAL)"
                    elif token in detector.FAKE_TOKENS:
                        interpretation = "(FAKE)"

                    assistant_msg += f"\n{i}. `{token}`: {logprob:.3f} → {prob:.4f} {interpretation}"

                assistant_msg += f"""

**Softmax Normalized:**
- P(Fake) = {result['confidence']:.4f} ({confidence_pct:.1f}%)
- P(Real) = {(1 - result['confidence']):.4f} ({(1 - result['confidence'])*100:.1f}%)

**Three-Tier Classification:**
- Tier: **{tier}**
- Threshold Check:
  * P_fake < 0.50? {'YES → Authentic' if result['confidence'] < 0.50 else 'NO'}
  * P_fake ≥ 0.90? {'YES → Deepfake' if result['confidence'] >= 0.90 else 'NO'}

**Verdict Token:** `{result['verdict_token']}`

---
### ⚙️ System Prompt

```
{debug['system_prompt']}
```

---
### ⏱️ Performance Metrics

**Stage-by-Stage Timing:**
- Stage 0 (Metadata): {debug.get('stage_0_time', 0):.2f}s
- Stage 1 (Forensics): {debug.get('stage_1_time', 0):.2f}s
- Stage 2 (VLM Analysis): {debug['request_1_latency']:.2f}s
- Stage 3 (Verdict): {debug['request_2_latency']:.2f}s

**Total Pipeline:** {debug['total_pipeline_time']:.2f}s

**KV-Cache Hit:** {'✅ YES' if debug.get('kv_cache_hit', False) else '❌ NO'}
"""

            st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
            st.rerun()

        except Exception as e:
            error_msg = f"❌ **Error during detection:**\n```\n{str(e)}\n```"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.error(f"Detection failed: {str(e)}")
```

## 5. Update Forensic Artifacts Display (Lines 120-175)

Replace the forensic result display section with:

```python
# Show detailed forensic result if available
if st.session_state.osint_result is not None:
    result = st.session_state.osint_result
    st.markdown("---")
    st.markdown("### OSINT Detection Result")

    tier = result['tier']
    confidence = result['confidence']

    # Visual confidence bar with color coding
    if tier == "Deepfake":
        st.error(f"🚨 **{tier}** - Confidence: {confidence*100:.1f}%")
    elif tier == "Suspicious":
        st.warning(f"⚠️ **{tier}** - Confidence: {confidence*100:.1f}%")
    else:
        st.success(f"✅ **{tier}** - Confidence: {confidence*100:.1f}%")

    st.progress(confidence, text=f"P(Fake): {confidence*100:.1f}%")

    # Metadata auto-fail indicator
    if result.get('metadata_auto_fail', False):
        st.error("⚠️ AI tool signature detected in metadata - Instant rejection")
```

## 6. Add Missing Import at Top

At the very top after other imports, add:

```python
import math  # For debug mode logprob calculations
```

## Testing Checklist

After making these changes:

1. ✅ Test with image upload (auto context)
2. ✅ Test with each OSINT context (military/disaster/propaganda)
3. ✅ Toggle debug mode on/off
4. ✅ Verify forensic artifacts display correctly
5. ✅ Check debug panels expand/collapse
6. ✅ Test with AI-generated image (should auto-fail if metadata present)
7. ✅ Verify performance metrics are accurate
8. ✅ Check KV-cache optimization is working (Request 2 should be <0.5s)

## Notes

- The old `classifier.py` code remains functional for Tab 2 (batch evaluation)
- This update is backward compatible - Tab 2 continues using the legacy classifier
- All new functionality is isolated to Tab 1
- Debug mode can be toggled without re-uploading images (just rerun detection)
