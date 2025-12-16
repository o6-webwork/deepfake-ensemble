# Docker Quick Start Guide - Phase 1 Testing

Quick guide to test the OSINT detection system in a Docker container.

---

## Prerequisites

✅ Docker installed and running
✅ Docker Compose v2 (comes with modern Docker)
✅ At least one VLM model server running (see config.py for URLs)

---

## Quick Start (3 commands)

```bash
# 1. Build the container
docker compose build

# 2. Start the application
docker compose up -d

# 3. Check logs
docker compose logs -f
```

**Access the app:** http://localhost:8501

---

## What's Included in the Container

The Dockerfile includes all Phase 1 components:

✅ `app.py` - Streamlit UI with OSINT controls
✅ `detector.py` - OSINT detection pipeline (NEW)
✅ `forensics.py` - Enhanced forensics (metadata + FFT preprocessing)
✅ `classifier.py` - Legacy classifier (for Tab 2)
✅ `config.py` - Model configurations
✅ `shared_functions.py` - Utility functions
✅ `requirements.txt` - Including `exifread>=3.0.0`

**Dependencies installed:**
- streamlit
- pillow
- opencv-python-headless
- numpy
- pandas
- openai
- xlsxwriter
- openpyxl
- **exifread** (NEW - for metadata extraction)

---

## Testing the Container

### 1. Verify Container is Running

```bash
docker compose ps
```

Expected output:
```
NAME                      STATUS              PORTS
deepfake-detector-app     Up X minutes        0.0.0.0:8501->8501/tcp
```

### 2. Check Health Status

```bash
docker compose exec deepfake-detector curl http://localhost:8501/_stcore/health
```

Should return: `ok`

### 3. View Live Logs

```bash
docker compose logs -f deepfake-detector
```

You should see:
```
You can now view your Streamlit app in your browser.
Local URL: http://0.0.0.0:8501
Network URL: http://172.x.x.x:8501
```

---

## Testing OSINT Features in Container

### Basic Test Flow

1. **Open browser:** http://localhost:8501

2. **Upload an image** (jpg, png, or video)

3. **Select OSINT Context:**
   - Auto-Detect (default)
   - Military
   - Disaster
   - Propaganda

4. **Enable Debug Mode** (checkbox)

5. **Wait for detection pipeline** (~3-5 seconds)

6. **Verify output:**
   - Three-tier classification (Authentic/Suspicious/Deepfake)
   - VLM reasoning displayed
   - Forensic report with EXIF, ELA, FFT metrics
   - If debug enabled: 6 debug sections appear

### Test Cases

**Test 1: Real Photo**
```bash
# Upload a real photograph
# Expected: "Authentic" or "Suspicious" (depending on post-processing)
# ELA variance: Should vary, check threshold
# FFT pattern: "Chaotic" (natural)
```

**Test 2: AI-Generated Image with Metadata**
```bash
# Upload an image from Midjourney/Stable Diffusion (with EXIF)
# Expected: Instant "Deepfake" (metadata auto-fail)
# Message: "AI generation tool detected in metadata"
```

**Test 3: Panorama (FFT Center Crop Test)**
```bash
# Upload a wide panorama (e.g., 3840x1080)
# Expected: FFT patterns should NOT be distorted
# Check: FFT should show correct Grid/Chaotic classification
```

**Test 4: Debug Mode**
```bash
# Enable debug mode checkbox
# Upload any image
# Expected: See all 6 debug sections:
#   1. Forensic Lab Report
#   2. VLM Analysis Output
#   3. Logprobs & Verdict (Top K=5 tokens)
#   4. System Prompt
#   5. Performance Metrics
#   6. KV-Cache status
```

**Test 5: OSINT Contexts**
```bash
# Upload military formation image
# Select "Military" context
# Expected: System prompt mentions "CASE A: Military Context"
# Check: FFT threshold adjustment mentioned (+20%)

# Upload disaster/flood image
# Select "Disaster" context
# Expected: System prompt mentions "CASE B: Disaster Context"
# Check: High-entropy noise filtering mentioned

# Upload studio portrait
# Select "Propaganda" context
# Expected: System prompt mentions "CASE C: Propaganda Context"
# Check: ELA contrast expectations mentioned
```

---

## Performance Benchmarks (In Container)

Check debug mode output for these metrics:

| Metric | Expected | Location |
|--------|----------|----------|
| Total Pipeline | 3-5s | Performance Metrics section |
| Stage 0 (Metadata) | <0.1s | Performance Metrics |
| Stage 1 (Forensics) | ~1s | Performance Metrics |
| Stage 2 (Analysis) | 2-3s | API Metadata (Request 1 Latency) |
| Stage 3 (Verdict) | <0.5s | API Metadata (Request 2 Latency) |
| KV-Cache Improvement | >85% | Calculated from Request 1 vs 2 |
| Request 2 Tokens | ~10 | API Metadata |

**KV-Cache Verification:**
- Request 2 should show "⚡ XX% faster via KV-cache"
- Should be >80% faster than Request 1
- KV-Cache Hit: Should show "✅ YES"

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker compose logs deepfake-detector

# Common issues:
# 1. Port 8501 already in use
#    Solution: Stop other Streamlit instances or change port in docker-compose.yml

# 2. Build failed
#    Solution: Check requirements.txt has all dependencies
```

### "No module named 'detector'"

```bash
# Rebuild container with updated Dockerfile
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Model endpoint not reachable

```bash
# Check if model servers are accessible from container
docker compose exec deepfake-detector curl -I http://100.64.0.3:8006/v1/

# If you get connection refused:
# - Model servers need to be on same Docker network, OR
# - Use host.docker.internal instead of localhost in config.py, OR
# - Use --network host in docker-compose.yml (Linux only)
```

### EXIF metadata not extracted

```bash
# Verify exifread is installed
docker compose exec deepfake-detector pip list | grep exifread

# Should show: exifread  3.0.0 (or higher)

# If missing, rebuild:
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Debug mode not showing all sections

- Check that image was uploaded after enabling debug mode
- Try toggling debug mode off and on
- Refresh browser page
- Check container logs for errors

---

## Useful Commands

### View Container Resources

```bash
# CPU/Memory usage
docker stats deepfake-detector-app
```

### Exec into Container

```bash
# Open shell in container
docker compose exec deepfake-detector /bin/bash

# Check files
ls -la /app

# Test Python imports
python3 -c "from detector import OSINTDetector; print('OK')"
```

### Restart Container

```bash
# Restart (keeps data)
docker compose restart

# Full restart (rebuilds if needed)
docker compose down
docker compose up -d
```

### Stop and Clean Up

```bash
# Stop containers
docker compose down

# Stop and remove volumes
docker compose down -v

# Remove images too
docker compose down --rmi all
```

---

## Container Configuration

### Resource Limits (in docker-compose.yml)

Current settings:
- **CPU Limit:** 2.0 cores
- **Memory Limit:** 4GB
- **CPU Reservation:** 1.0 core
- **Memory Reservation:** 2GB

Adjust if needed based on your system.

### Volumes Mounted

These directories are mounted for persistence:
- `./results` → Evaluation results
- `./testing_files` → Test images
- `./misc` → Miscellaneous files
- `./analysis_output` → Analysis outputs

Files uploaded via Streamlit are stored in container memory (not persisted).

### Network

- Container runs on `deepfake-network` (bridge mode)
- Port 8501 mapped to host
- Can communicate with other containers on same network

---

## Production Considerations

### Security (Already Implemented)

✅ **Non-root user:** Container runs as UID 1001 (appuser)
✅ **No unnecessary packages:** Minimal base image
✅ **Health checks:** Built-in health monitoring
✅ **No secrets in image:** API keys should be in environment variables

### Additional Hardening (Optional)

```yaml
# Add to docker-compose.yml:
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE  # Only if needed
```

### Environment Variables (for Production)

```yaml
environment:
  # Model endpoints (if different from config.py)
  - MODEL_QWEN3_URL=http://your-model-server:8006/v1/
  - MODEL_QWEN3_KEY=your-api-key-here

  # Streamlit config
  - STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
  - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200  # MB
```

---

## Quick Test Script

Save as `test_container.sh`:

```bash
#!/bin/bash

echo "🐳 Testing OSINT Detection Container"
echo "===================================="

# Check if container is running
if ! docker compose ps | grep -q "Up"; then
    echo "❌ Container not running. Starting..."
    docker compose up -d
    sleep 5
fi

# Health check
echo "1. Health Check..."
HEALTH=$(docker compose exec -T deepfake-detector curl -s http://localhost:8501/_stcore/health)
if [ "$HEALTH" = "ok" ]; then
    echo "   ✅ Health: OK"
else
    echo "   ❌ Health: FAILED"
    exit 1
fi

# Check detector module
echo "2. Checking detector.py import..."
DETECTOR=$(docker compose exec -T deepfake-detector python3 -c "from detector import OSINTDetector; print('OK')" 2>&1)
if echo "$DETECTOR" | grep -q "OK"; then
    echo "   ✅ Detector: OK"
else
    echo "   ❌ Detector: FAILED"
    echo "   $DETECTOR"
    exit 1
fi

# Check exifread
echo "3. Checking exifread dependency..."
EXIFREAD=$(docker compose exec -T deepfake-detector pip list | grep exifread)
if [ -n "$EXIFREAD" ]; then
    echo "   ✅ exifread: $EXIFREAD"
else
    echo "   ❌ exifread: NOT INSTALLED"
    exit 1
fi

echo ""
echo "✅ All checks passed!"
echo "🌐 Access app at: http://localhost:8501"
echo ""
echo "📋 Test checklist:"
echo "   1. Upload an image"
echo "   2. Select OSINT context"
echo "   3. Enable debug mode"
echo "   4. Verify 3-tier classification"
echo "   5. Check KV-cache optimization (Request 2 < 0.5s)"
```

Make executable and run:
```bash
chmod +x test_container.sh
./test_container.sh
```

---

## Success Criteria

Container deployment is successful if:

✅ Container builds without errors
✅ Health check returns "ok"
✅ App loads at http://localhost:8501
✅ OSINT context selector visible
✅ Debug mode toggle works
✅ Image upload triggers detection
✅ Three-tier classification displays
✅ Debug mode shows all 6 sections
✅ KV-cache reduces Request 2 latency by >80%
✅ Performance metrics match benchmarks

---

## Next Steps After Container Testing

1. **If all tests pass:** System ready for production
2. **If tests fail:** Check troubleshooting section above
3. **For Phase 2:** Model management UI (not yet containerized)
4. **For Phase 3:** Batch evaluation updates

---

**Quick Reference:**

```bash
# Start
docker compose up -d

# Test
open http://localhost:8501

# Logs
docker compose logs -f

# Stop
docker compose down
```

Happy testing! 🚀
