# Docker Testing Guide - Phase 1 Forensic System

**Branch:** `feature/forensic-scanner-and-model-management`
**Commit:** `26c7db2`
**Date:** December 11, 2025

---

## Overview

This guide walks through manual Docker testing of the Phase 1 forensic detection system, validating that ELA/FFT generation and logit calibration work correctly in the containerized environment.

---

## Prerequisites

### 1. Model Endpoints Running

Ensure your vLLM model servers are running and accessible:

```bash
# Test model connectivity
curl http://100.64.0.1:8000/v1/models        # InternVL 2.5 8B
curl http://localhost:1234/v1/models         # InternVL 3.5 8B
curl http://100.64.0.3:8001/v1/models        # MiniCPM-V 4.5
curl http://100.64.0.3:8006/v1/models        # Qwen3 VL 32B
```

Expected response: JSON with model information

If models aren't running, the forensic system will still generate ELA/FFT but classification will fail.

### 2. Test Images Available

Ensure you have test images in `testing_files/`:

```bash
ls testing_files/Curated\ Real/
ls testing_files/Curated\ AI-Generated/
ls testing_files/Curated\ AI-Manipulated/
```

---

## Testing Procedure

### Step 1: Build Docker Image

```bash
cd "/home/otb-02/Desktop/deepfake detection"

# Build with new forensic modules
docker compose build

# Or force rebuild without cache
docker compose build --no-cache
```

**Expected Output:**
```
[+] Building X.Xs (XX/XX) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: XXB
 => [internal] load .dockerignore
 => [internal] load metadata for docker.io/library/python:3.11-slim
 ...
 => exporting to image
 => => exporting layers
 => => writing image sha256:...
 => => naming to docker.io/library/deepfake-detection-deepfake-detector
```

**Verify Files Copied:**
Look for these lines in build output:
```
COPY --chown=appuser:appuser forensics.py .
COPY --chown=appuser:appuser classifier.py .
```

### Step 2: Start Container

```bash
docker compose up -d

# Check container status
docker compose ps
```

**Expected Output:**
```
NAME                        IMAGE                              STATUS
deepfake-detector-app       deepfake-detection-deepfake-detector   Up X seconds
```

**Check Logs:**
```bash
docker compose logs -f
```

**Expected in logs:**
```
You can now view your Streamlit app in your browser.
Local URL: http://0.0.0.0:8501
Network URL: http://172.X.X.X:8501
```

Press `Ctrl+C` to exit log view.

### Step 3: Access Web Interface

Open browser to: **http://localhost:8501**

**Expected:**
- Streamlit app loads successfully
- Two tabs visible: "🔍 Detection" and "📊 Evaluation"
- Model selector dropdown with 4 models
- File uploader visible

---

## Test Cases

### Test Case 1: Forensic Module Import

**Objective:** Verify forensics.py and classifier.py load without errors

**Steps:**
```bash
# Enter container
docker compose exec deepfake-detector /bin/bash

# Test forensics module
python3 -c "from forensics import generate_both; print('✓ forensics.py imported')"

# Test classifier module
python3 -c "from classifier import ForensicClassifier; print('✓ classifier.py imported')"

# Exit container
exit
```

**Expected Output:**
```
✓ forensics.py imported
✓ classifier.py imported
```

**If errors occur:**
- Check that opencv-python-headless is installed: `pip list | grep opencv`
- Check Python version: `python3 --version` (should be 3.11)

---

### Test Case 2: ELA Generation

**Objective:** Verify Error Level Analysis generation works correctly

**Steps:**
```bash
# Enter container
docker compose exec deepfake-detector /bin/bash

# Test ELA on a real image
python3 forensics.py testing_files/Curated\ Real/HADR/HADR_REAL_DAY_FLOOD_01.jpg

# Test ELA on an AI image
python3 forensics.py testing_files/Curated\ AI-Generated/HADR/HADR_AIG_DAY_FLOOD_01.jpg

# Exit container
exit
```

**Expected Output:**
```
Generating forensic artifacts for: testing_files/Curated Real/HADR/HADR_REAL_DAY_FLOOD_01.jpg
✓ ELA map saved to: testing_files/Curated Real/HADR/HADR_REAL_DAY_FLOOD_01_ela.png
✓ FFT spectrum saved to: testing_files/Curated Real/HADR/HADR_REAL_DAY_FLOOD_01_fft.png

Forensic Analysis:
  - Check ELA map: Uniform rainbow = AI, Dark with edges = Real
  - Check FFT spectrum: Grid/Cross = AI, Chaotic starburst = Real
```

**Validation:**
Files should be created in the `testing_files/` directory.

**Visual Inspection:**
- **ELA (Real):** Should show dark regions with bright edges (compression varies)
- **ELA (AI):** Should show uniform bright patterns (compression consistent)
- **FFT (Real):** Should show chaotic starburst pattern
- **FFT (AI):** Should show grid or cross patterns

---

### Test Case 3: Web UI - Upload and Analyze

**Objective:** Test full forensic classification pipeline through UI

**Steps:**

1. **Navigate to Detection Tab:**
   - Open http://localhost:8501
   - Ensure "🔍 Detection" tab is selected

2. **Upload Real Image:**
   - Click file uploader
   - Select: `testing_files/Curated Real/HADR/HADR_REAL_DAY_FLOOD_01.jpg`
   - Wait for "🔬 Generating forensic artifacts..." spinner

3. **Verify Forensic Analysis:**
   - Check chat panel shows:
     - **Result:** Should be "Authentic" (for real image)
     - **Confidence:** Should be < 50% (low AI probability)
     - **Raw Logprobs:** P(FAKE) and P(REAL) values shown
     - **Token Output:** Should be "REAL" or similar

4. **View Forensic Artifacts:**
   - Click "🔬 View Forensic Artifacts" expander in left panel
   - Verify two images displayed:
     - **ELA Map:** Shows compression analysis
     - **FFT Spectrum:** Shows frequency analysis
   - Check captions explain AI vs Real signatures

5. **Check Confidence Visualization:**
   - Progress bar showing AI confidence percentage
   - Classification indicator (✅ for Authentic)
   - Confidence interpretation message

6. **Upload AI-Generated Image:**
   - Upload: `testing_files/Curated AI-Generated/HADR/HADR_AIG_DAY_FLOOD_01.jpg`
   - Verify:
     - **Result:** Should be "AI-Generated"
     - **Confidence:** Should be > 50% (high AI probability)
     - **Token Output:** Should be "FAKE" or similar

7. **Upload AI-Manipulated Image:**
   - Upload: `testing_files/Curated AI-Manipulated/HADR/HADR_AIM_DAY_STORM_01.jpg`
   - Verify classification (may vary based on manipulation type)

**Expected Behavior:**
- No errors in browser console (F12 Developer Tools)
- Forensic artifacts generate within 5-10 seconds
- Classification completes within 10-30 seconds (depends on model)
- Chat history preserved across uploads

---

### Test Case 4: Error Handling - Model Offline

**Objective:** Verify graceful fallback when model endpoint unavailable

**Steps:**

1. **Temporarily Stop Model Server** (or use invalid endpoint):
   ```bash
   # Option 1: Modify config.py temporarily in container
   docker compose exec deepfake-detector /bin/bash
   # Edit config.py base_url to point to non-existent endpoint

   # Option 2: Test with network disconnected
   ```

2. **Upload Image:**
   - Upload any test image
   - Observe error handling

**Expected Behavior:**
- Error message displayed: "Error during forensic analysis: ..."
- System falls back to standard analysis (old method)
- OR shows error message with neutral 0.5 confidence

**Cleanup:**
- Restore original config.py or restart container
- `docker compose restart`

---

### Test Case 5: Multiple Models

**Objective:** Test forensic classification with all 4 models

**Steps:**

1. **Test with InternVL 2.5 8B:**
   - Select "InternVL 2.5 8B" from model dropdown
   - Upload test image
   - Record confidence score

2. **Test with InternVL 3.5 8B:**
   - Select "InternVL 3.5 8B"
   - Upload same test image
   - Record confidence score

3. **Test with MiniCPM-V 4.5:**
   - Select "MiniCPM-V 4.5"
   - Upload same test image
   - Record confidence score

4. **Test with Qwen3 VL 32B:**
   - Select "Qwen3 VL 32B"
   - Upload same test image
   - Record confidence score

**Analysis:**
- Compare confidence scores across models
- Verify all models complete successfully
- Note if certain models are more confident
- Expected: Qwen3 VL 32B should have best calibration

---

## Validation Checklist

### ✅ Forensic Artifact Generation

- [ ] ELA maps generate successfully
- [ ] FFT spectra generate successfully
- [ ] Both artifacts visible in UI expander
- [ ] Artifacts look visually distinct for AI vs Real images

**Visual Validation:**
- [ ] Real images: ELA shows dark regions, FFT shows starburst
- [ ] AI images: ELA shows uniform patterns, FFT shows grid/cross

### ✅ Classification Results

- [ ] Real images classified as "Authentic" (majority)
- [ ] AI-Generated images classified as "AI-Generated" (majority)
- [ ] AI-Manipulated images classified appropriately
- [ ] Confidence scores range from 0.0 to 1.0
- [ ] Token output is "REAL" or "FAKE" (single token)

### ✅ UI Components

- [ ] Original image displays correctly
- [ ] Forensic artifacts expander opens/closes
- [ ] ELA and FFT images display side-by-side
- [ ] Progress bar shows confidence percentage
- [ ] Classification indicator shows correct result
- [ ] Raw logprob values displayed
- [ ] Chat history preserved

### ✅ Performance

- [ ] Forensic artifacts generate in < 10 seconds
- [ ] Classification completes in < 30 seconds
- [ ] No memory leaks (test with 5+ uploads)
- [ ] Container remains responsive

### ✅ Error Handling

- [ ] Invalid images show error message
- [ ] Model offline triggers graceful fallback
- [ ] No unhandled exceptions in logs
- [ ] Error messages are user-friendly

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Cause:** opencv-python-headless not installed

**Solution:**
```bash
docker compose exec deepfake-detector pip list | grep opencv
# Should show: opencv-python-headless 4.8.0 (or higher)

# If missing, rebuild container
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Issue: "Connection refused" when classifying

**Cause:** Model endpoint not accessible from container

**Solution:**
```bash
# Test connectivity from inside container
docker compose exec deepfake-detector curl http://100.64.0.1:8000/v1/models

# Check docker network settings
docker compose exec deepfake-detector ping -c 2 100.64.0.1

# Verify model servers are running on host
curl http://100.64.0.1:8000/v1/models
```

### Issue: ELA/FFT images not displaying

**Cause:** Image bytes not converting correctly

**Solution:**
```bash
# Check logs for errors
docker compose logs | grep -i error

# Verify PIL/OpenCV interaction
docker compose exec deepfake-detector python3 -c "
from PIL import Image
import io
import cv2
import numpy as np
print('PIL:', Image.__version__)
print('OpenCV:', cv2.__version__)
"
```

### Issue: Container exits immediately

**Cause:** Syntax error or missing dependency

**Solution:**
```bash
# Check logs
docker compose logs deepfake-detector

# Try running directly
docker compose exec deepfake-detector streamlit run app.py --server.port=8501

# Validate Python syntax
docker compose exec deepfake-detector python3 -m py_compile app.py
docker compose exec deepfake-detector python3 -m py_compile forensics.py
docker compose exec deepfake-detector python3 -m py_compile classifier.py
```

---

## Performance Benchmarks

### Expected Timings

| Operation | Expected Time | Acceptable Range |
|-----------|---------------|------------------|
| Forensic artifact generation | 2-5 seconds | < 10 seconds |
| VLM classification (single image) | 5-15 seconds | < 30 seconds |
| Total (upload → result) | 10-20 seconds | < 40 seconds |

### Resource Usage

Monitor with:
```bash
docker stats deepfake-detector-app
```

**Expected:**
- CPU: 50-100% during processing, <5% idle
- Memory: 500MB-2GB (depends on image size)
- Should stay under 4GB limit (docker-compose.yml)

---

## Success Criteria

Phase 1 is validated if:

1. ✅ **Forensic modules import successfully**
2. ✅ **ELA/FFT generate without errors**
3. ✅ **Real images show correct forensic patterns**
4. ✅ **AI images show correct forensic patterns**
5. ✅ **Classification works with at least 1 model**
6. ✅ **UI displays all forensic components**
7. ✅ **Confidence scores are in [0.0, 1.0] range**
8. ✅ **No unhandled exceptions occur**

**Bonus:**
- All 4 models work correctly
- Accuracy > 55.56% baseline (requires full dataset test)
- Confidence scores are well-calibrated

---

## Next Steps After Validation

### If All Tests Pass ✅

1. **Document Results:**
   - Screenshot forensic artifacts (Real vs AI comparison)
   - Record confidence scores for sample images
   - Note any performance observations

2. **Proceed to Batch Testing:**
   - Run full evaluation on 72-image dataset (Tab 2)
   - Compare accuracy: old system vs forensic system
   - Generate calibration plots

3. **Optimize Thresholds:**
   - Analyze confidence score distribution
   - Determine optimal threshold per model
   - Update `CALIBRATION_CONFIG` in unified design spec

### If Tests Fail ❌

1. **Identify Failure Point:**
   - Module import? → Check dependencies
   - ELA/FFT generation? → Debug forensics.py
   - Classification? → Check model connectivity
   - UI display? → Check Streamlit logs

2. **Fix and Retest:**
   - Make necessary corrections
   - Rebuild container
   - Re-run failed test cases

3. **Report Issues:**
   - Document error messages
   - Capture logs: `docker compose logs > debug.log`
   - Check browser console errors (F12)

---

## Clean Up

After testing:

```bash
# Stop container
docker compose down

# Remove volumes (optional, clears results)
docker compose down -v

# Remove images (optional, forces full rebuild)
docker rmi deepfake-detection-deepfake-detector
```

---

**Testing Status:** Ready for manual validation
**Next Action:** Execute test cases and validate Phase 1 implementation
**Documentation:** [PHASE1_PROGRESS.md](PHASE1_PROGRESS.md) for detailed implementation info
