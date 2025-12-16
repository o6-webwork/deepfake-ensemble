# Manual Analysis Button Feature

**Date:** December 15, 2025
**Feature:** Manual "Analyze" button with model switching support

---

## Overview

The UI now requires manual triggering of analysis via an "Analyze Image" button, replacing the previous auto-analysis-on-upload behavior. This provides:

1. **User control** - Analysis runs only when requested
2. **Model switching** - Re-analyze same image with different models
3. **Cost control** - Avoid accidental API calls to cloud providers
4. **Better UX** - Clear separation between upload and analysis steps

---

## Changes

### Before (Auto-Analysis)

```
User uploads image
    ↓
Analysis runs automatically
    ↓
Results displayed
```

**Issues:**
- ❌ No user control over when analysis runs
- ❌ Can't switch models without re-uploading
- ❌ Accidental cloud API calls cost money
- ❌ Confusing for users (analysis starts immediately)

### After (Manual Analysis)

```
User uploads image
    ↓
Image displayed (ready state)
    ↓
User selects model (can change)
    ↓
User clicks "Analyze Image" button
    ↓
Analysis runs
    ↓
Results displayed
```

**Benefits:**
- ✅ Full user control
- ✅ Model switching without re-upload
- ✅ Cost control (cloud APIs)
- ✅ Clear user flow

---

## UI Layout

```
┌─────────────────────────────────────────────────────────┐
│ 🕵️‍♂️ Deepfake Detection Chat                              │
├─────────────────────────────────────────────────────────┤
│ Select detection model: [Qwen3 VL 32B ▼]               │
│                                                         │
│ OSINT Context: [Auto-Detect ▼]                         │
│                                                         │
│ ☑️ 🔍 Enable Debug Mode                                  │
│                                                         │
│ ┌───────────────────────────────────────────────────┐ │
│ │  🔍 Analyze Image                                  │ │  ← NEW BUTTON
│ └───────────────────────────────────────────────────┘ │
│                                                         │
│ ▼ ⚙️ Advanced Settings                                  │
│   ┌───────────────────────────────────────────────┐   │
│   │ ☐ 📊 Send Forensic Report to VLM              │   │
│   │ 🏷️ Watermark Handling: [Ignore ▼]              │   │
│   └───────────────────────────────────────────────┘   │
│                                                         │
│ ┌─────────────┬─────────────────────────────────────┐ │
│ │ Left Panel  │ Right Panel (Chat)                  │ │
│ │             │                                     │ │
│ │ [Image]     │ Upload image/video                  │ │
│ │             │ [Upload button]                     │ │
│ │ [ELA/FFT]   │                                     │ │
│ │             │ Chat messages...                    │ │
│ └─────────────┴─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Button Behavior

### States

| State | Button Text | Enabled | Description |
|-------|------------|---------|-------------|
| **No Image** | "🔍 Analyze Image" | ❌ Disabled | No image uploaded yet |
| **Image Ready** | "🔍 Analyze Image" | ✅ Enabled | Image uploaded, ready to analyze |
| **Analyzing** | "🔍 Analyze Image" | ❌ Disabled (spinner shown) | Analysis in progress |
| **Analysis Complete** | "🔍 Analyze Image" | ✅ Enabled | Can re-analyze with different settings |

### Visual Design

```python
analyze_button = st.button(
    "🔍 Analyze Image",
    type="primary",  # Blue primary button
    disabled=(st.session_state.media is None),  # Disabled if no image
    use_container_width=True,  # Full width
    help="Run deepfake detection on the uploaded image"
)
```

---

## Workflow Examples

### Example 1: First-Time Analysis

1. **Upload image**: `military_parade.jpg`
   - Image displayed in left panel
   - Analyze button becomes **enabled**

2. **Select model**: Keep default "Qwen3 VL 32B"

3. **Configure settings** (optional):
   - OSINT Context: "Military"
   - Send Forensics: Disabled
   - Watermark: Ignore

4. **Click "Analyze Image"**
   - Spinner shows: "🔬 Running OSINT detection pipeline..."
   - Progress: Stage 0 → Stage 1 → Stage 2 → Stage 3
   - Results appear in chat panel

5. **View results**:
   - Classification: Authentic
   - AI Generated Probability: 5.2%
   - VLM Reasoning: "No AI artifacts detected..."

### Example 2: Re-Analysis with Different Model

1. **Previous analysis complete** (image still loaded)

2. **Change model**: Select "OpenAI GPT-4o (Cloud)"

3. **Click "Analyze Image" again**
   - New analysis runs with GPT-4o
   - Previous result remains in chat history
   - New result appended as separate collapsible block

4. **Compare results**:
   - Qwen3 VL: 5.2% AI Generated
   - GPT-4o: 8.7% AI Generated
   - User can see reasoning from both models

### Example 3: Testing Different Settings

Same image, same model, different settings:

**Run 1:** With forensics enabled
```
Settings:
- Send Forensics: ✅ Enabled
- Result: Suspicious (52% AI Generated)
- Reason: "FFT shows grid artifacts..."
```

**Run 2:** With forensics disabled
```
Settings:
- Send Forensics: ❌ Disabled
- Result: Authentic (12% AI Generated)
- Reason: "Visual analysis shows natural textures..."
```

Demonstrates that forensics can introduce noise (as user suspected).

---

## Implementation

### Code Changes

**[app.py](app.py#L89-L95)** - Added Analyze button:
```python
# Analyze Button
analyze_button = st.button(
    "🔍 Analyze Image",
    type="primary",
    disabled=(st.session_state.media is None),
    use_container_width=True,
    help="Run deepfake detection on the uploaded image"
)
```

**[app.py](app.py#L132-L147)** - Disabled auto-analysis on upload:
```python
# Before:
st.session_state.messages.append(
    {"role": "user", "content": f"📷 Uploaded image: {uploaded_file.name}"}
)

# After:
# Don't auto-append message anymore - wait for Analyze button
```

**[app.py](app.py#L274-L286)** - Trigger analysis on button click:
```python
# Run analysis when Analyze button is clicked (not on upload)
if analyze_button and analysis_image:
    with st.spinner("🔬 Running OSINT detection pipeline..."):
        # Add upload message if this is first analysis
        if uploaded_file and not any(...):
            st.session_state.messages.append(
                {"role": "user", "content": f"{file_type}: {uploaded_file.name}"}
            )
        # ... rest of analysis logic
```

---

## Use Cases

### Use Case 1: Cost-Conscious Cloud Usage

**Scenario:** User wants to test image before committing to expensive GPT-4o analysis

**Workflow:**
1. Upload image
2. First run: Use free Gemini 2.0 Flash
3. If suspicious: Re-run with GPT-4o for higher confidence
4. If clearly authentic: Skip expensive model

**Benefit:** Save ~$0.04 per image by pre-filtering

### Use Case 2: Model Comparison Research

**Scenario:** Researcher comparing VLM capabilities

**Workflow:**
1. Upload test image
2. Run with Qwen3 VL (local)
3. Run with GPT-4o (cloud)
4. Run with Claude 3.5 Sonnet (cloud)
5. Run with Gemini 2.0 Flash (cloud)
6. Compare all 4 results side-by-side in chat history

**Benefit:** Easy A/B testing without re-uploading

### Use Case 3: Settings Experimentation

**Scenario:** User testing impact of forensic toggle

**Workflow:**
1. Upload image
2. Run with forensics enabled
3. Run with forensics disabled
4. Compare reasoning to see if forensics add signal or noise

**Benefit:** Understand which settings work best for their use case

### Use Case 4: Batch Processing Workflow

**Scenario:** Analyst reviewing 50 images from a campaign

**Workflow:**
1. Upload image 1
2. Review image visually first
3. If suspicious → Click Analyze
4. If obviously real → Skip analysis, upload next
5. Repeat for all 50 images

**Benefit:** Save API calls on obviously authentic images

---

## User Experience Improvements

### Before (Auto-Analysis)

**User uploads image**
```
User: [uploads military_parade.jpg]
Assistant: 📷 Uploaded image: military_parade.jpg
Assistant: [119 seconds of waiting...]
Assistant: Classification: Authentic...
```

**Problems:**
- User has 119 seconds of dead time
- Can't change settings mid-analysis
- Can't cancel if wrong model selected

### After (Manual Analysis)

**User uploads image**
```
User: [uploads military_parade.jpg]
[Image appears in left panel]
[Analyze button enabled]

User: [selects GPT-4o, configures settings]
User: [clicks Analyze Image]
Assistant: 📷 Uploaded image: military_parade.jpg
Assistant: [15 seconds of waiting...]
Assistant: Classification: Authentic...
```

**Benefits:**
- User controls when analysis starts
- Can review settings before running
- Much faster with cloud models (15s vs 119s)
- Clear visual feedback (button state)

---

## Accessibility

### Keyboard Navigation

```
Tab → Focus model dropdown
Tab → Focus OSINT context dropdown
Tab → Focus debug mode checkbox
Tab → Focus Analyze button
Enter → Trigger analysis
```

### Screen Readers

```html
<button
    aria-label="Analyze uploaded image for deepfake detection"
    disabled="true|false"
>
    🔍 Analyze Image
</button>
```

### Visual Indicators

| State | Color | Icon | Help Text |
|-------|-------|------|-----------|
| Enabled | Blue (primary) | 🔍 | "Run deepfake detection on the uploaded image" |
| Disabled | Gray | 🔍 | "Upload an image first" |
| Loading | Blue + Spinner | 🔬 | "Running OSINT detection pipeline..." |

---

## Error Handling

### Scenario 1: No Image Uploaded

**Before:** N/A (auto-analysis wouldn't trigger)

**After:**
- Button is **disabled** (grayed out)
- Tooltip shows: "Upload an image first"
- User cannot accidentally click

### Scenario 2: API Key Missing (Cloud Model)

**Before:** Would fail after 119s of processing

**After:**
- Fails fast (<1s) with clear error
- Error message: "API key not found for OpenAI. Set OPENAI_API_KEY environment variable."
- User can fix and retry without re-uploading

### Scenario 3: Network Error (Cloud Model)

**Before:** Timeout after 180s

**After:**
- Fails at provider timeout (~60s)
- Error message shows in chat
- Forensic report still displayed (error handling)
- User can switch to local model and retry

---

## Performance Considerations

### State Management

```python
# Session state variables
st.session_state.media          # Uploaded image (persists)
st.session_state.last_uploaded  # Filename (prevents re-upload)
st.session_state.messages       # Chat history (persists)
st.session_state.osint_result   # Latest result (updates)
```

**Benefit:** Image stays in memory, no need to re-upload for model switching

### Caching

Streamlit's `@st.cache_data` is NOT used because:
- Each model produces different results
- Settings changes should trigger new analysis
- Cloud API results shouldn't be cached (cost tracking)

---

## Future Enhancements

### 1. Batch Analysis Button

```python
if st.button("🔍 Analyze All Uploaded Images"):
    for image in st.session_state.all_images:
        analyze(image)
```

### 2. Cancel Button

```python
col1, col2 = st.columns([3, 1])
with col1:
    analyze_button = st.button("🔍 Analyze Image")
with col2:
    if st.session_state.analyzing:
        cancel_button = st.button("❌ Cancel")
```

### 3. Keyboard Shortcut

```javascript
// Streamlit doesn't natively support this, but could add via custom component
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        triggerAnalysis();
    }
});
```

### 4. Auto-Save Results

```python
if analyze_button:
    result = detector.detect(...)
    # Auto-save to results/
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/analysis_{timestamp}.json', 'w') as f:
        json.dump(result, f)
```

---

## Related Files

- [app.py](app.py#L89-L95) - Analyze button implementation
- [app.py](app.py#L274-L286) - Button click handler
- [app.py](app.py#L132-L147) - Disabled auto-analysis on upload

---

## Sign-Off

**Feature:** Manual "Analyze Image" button
**Status:** ✅ Implementation complete
**Benefits:**
- User control over analysis timing
- Model switching without re-upload
- Cost control for cloud APIs
- Clearer user workflow
