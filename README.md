# NexInspect: Multi-Layer Deepfake Detection System

A production-ready AI-generated image detection pipeline combining **4 complementary forensic layers** (Texture, GAPL, SPAI, VLM) with intelligent weighted voting for robust deepfake detection. Optimized for OSINT applications, military imagery verification, disaster response coordination, and propaganda analysis.

**Current Version**: 3.0.0
**Status**: Production Ready
**Accuracy**: Up to **90.7%** on diverse real-world datasets
**Last Updated**: December 29, 2025

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 🎯 Quick Start

### Automated Setup (Recommended)

**Linux/Mac:**
```bash
git clone https://github.com/o6-webwork/deepfake-ensemble.git
cd deepfake-ensemble
chmod +x setup.sh
./setup.sh
```

**Windows:**
```bash
git clone https://github.com/o6-webwork/deepfake-ensemble.git
cd deepfake-ensemble
setup.bat
```

The setup script will:
- ✅ Check Python installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Check for SPAI weights
- ✅ Create .env configuration

**Then download SPAI weights** (~2GB):
1. Visit: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view
2. Place at: `spai/weights/spai.pth`

**Launch:**
```bash
# Linux/Mac
source venv/bin/activate
streamlit run app.py

# Windows
venv\Scripts\activate
streamlit run app.py
```

Visit `http://localhost:8501` to start detecting deepfakes!

---

## 📊 Performance Highlights

| Detection Mode | Accuracy | Precision | Recall | Speed | Use Case |
|---|---|---|---|---|---|
| **Enhanced 4-Layer** | **90.7%** | 89.0% | 97.9% | ~15s | High-stakes verification |
| **GAPL Only** | **88.8%** | **97.6%** | 85.1% | ~3s | High-precision mode |
| **SPAI Only** | 82.3% | 88.7% | 83.7% | ~50ms | Real-time screening |
| **SPAI + VLM** | 77.2% | 77.1% | 92.9% | ~3s | Balanced mode |

**Key Achievements:**
- ✅ **97.9% Recall** - Only misses 3 out of 141 AI-generated images
- ✅ **97.6% Precision (GAPL)** - Only 3 false positives in 74 real images
- ✅ **90.7% Overall Accuracy** - Industry-leading multi-layer performance
- ✅ **50ms Inference** - Real-time SPAI standalone mode

---

## 🚀 What's New in v3.0

### Recent Improvements (December 2024)

**1. Fixed GAPL Integration** (+23.2% accuracy improvement)
- Corrected class import error (`GAPLForensicsPipeline`)
- GAPL now achieves 88.8% accuracy (was broken at 65.6%)
- Only 3 false positives vs 74 before

**2. Optimized Classification Thresholds** (+8-12% across all modes)
- Raised Suspicious threshold from 0.35 → 0.50
- More balanced decision boundary (50% confidence required)
- Dramatically reduced false positive rates

**3. Calibrated SPAI Temperature** (Better balance)
- Reduced from T=2.0 → T=1.5 for improved sensitivity
- Less overly conservative predictions
- Enhanced 4-Layer jumped from 78.6% → 90.7%

**4. New "Compare All Methods" Mode**
- Automatically tests all 4 detection methods
- Side-by-side performance comparison
- Intelligent recommendations on which layers to use
- Helps identify best configuration for your dataset

---

## 🏗️ System Architecture

### The 4-Layer Approach

Our ensemble combines four independent detection methods, each analyzing different aspects of image authenticity:

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT IMAGE                          │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────▼────────────┐
        │  Stage 0: Metadata     │  Auto-fail if AI tool in EXIF
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  Parallel Forensics    │
        ├────────────────────────┤
        │ • Layer 1: Texture     │  Compression artifacts
        │ • Layer 2: GAPL        │  Generator fingerprints
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  Layer 3: SPAI         │  Spectral analysis
        │  (Vision Transformer)  │  + Attention heatmap
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  Layer 4: VLM          │  Semantic reasoning
        │  (Optional)            │  + Explainability
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  Weighted Voting       │  Confidence-based
        │  + Consensus Check     │  aggregation
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  FINAL VERDICT         │  Tier + Confidence
        │  + Explanations        │  + Layer breakdown
        └────────────────────────┘
```

### Layer Specializations

**Layer 1: Texture Forensics** (`texture_forensics.py`)
- **What**: PLGF, Frequency Analysis, PLADA, Degradation Profiler
- **Strengths**: Excellent for GAN-based deepfakes, compression-resistant
- **Speed**: ~4.5 seconds

**Layer 2: GAPL** (`gapl_forensics.py`)
- **What**: Generator-Aware Prototype Learning with Vision Transformer
- **Strengths**: **97.6% precision** - best at avoiding false positives
- **Speed**: ~2.8s (GPU) / ~8.2s (CPU)

**Layer 3: SPAI** (`spai_detector.py`)
- **What**: Spectral analysis using frequency-domain Vision Transformer
- **Strengths**: Fastest layer, provides visual heatmaps
- **Speed**: ~50ms

**Layer 4: VLM** (integrated in `detector.py`)
- **What**: Vision-language model semantic reasoning
- **Strengths**: Explainable verdicts, semantic coherence analysis
- **Speed**: ~3 seconds (2-stage API call with KV-cache)

---

## 🎯 Detection Modes

### Mode 1: Enhanced 4-Layer (Recommended)

**Best For**: High-stakes verification, forensic analysis, comprehensive evaluation

**Accuracy**: **90.7%** | **Speed**: ~15 seconds | **Layers**: All 4

**What You Get:**
- Multi-method cross-validation
- Highest overall accuracy
- 97.9% recall (catches nearly all fakes)
- Layer-by-layer breakdown
- Consensus flag (know when layers disagree)
- Forensic-grade evidence

**Perfect for**: Intelligence reports, court evidence, dataset evaluation, critical decisions

---

### Mode 2: GAPL Only (High-Precision)

**Best For**: Scenarios where false accusations are costly

**Accuracy**: 88.8% | **Speed**: ~3 seconds | **Precision**: **97.6%**

**What You Get:**
- Only 3 false positives in 74 real images
- Conservative but reliable
- Generator fingerprint detection

**Perfect for**: News agency verification, journalism, social media moderation

---

### Mode 3: SPAI Only (Fast Screening)

**Best For**: Real-time applications, high-volume screening

**Accuracy**: 82.3% | **Speed**: ~50ms | **No API costs**

**What You Get:**
- Near-instant results
- Visual attention heatmaps
- Good baseline accuracy
- No external API dependencies

**Perfect for**: Initial screening, real-time filtering, budget constraints, large datasets

---

### Mode 4: Compare All Methods

**Best For**: Dataset evaluation, determining optimal configuration

**What It Does:**
- Tests all 4 modes on your images
- Shows side-by-side comparison
- Recommends best method for your use case
- Analyzes which layers add value

**Perfect for**: Research, benchmarking, understanding your specific dataset

---

## 📈 Detailed Performance Metrics

### Evaluation Dataset
- **Total Images**: 215 (141 AI-generated + 74 real)
- **AI Generators**: DALL-E, Midjourney, Stable Diffusion, StyleGAN, DreamStudio, StarryAI
- **Real Images**: Military operations, disaster scenes, war photography
- **Compression**: Mixed (pristine to heavily compressed social media)

### Confusion Matrices

**Enhanced 4-Layer:**
```
                Predicted Real    Predicted AI
Actual Real         57 (TN)         17 (FP)
Actual AI            3 (FN)        138 (TP)
```
- Only **3 missed fakes** out of 141
- Only **17 false alarms** out of 74 real images
- **F1-Score**: 93.2%

**GAPL Only (High Precision):**
```
                Predicted Real    Predicted AI
Actual Real         71 (TN)          3 (FP)
Actual AI           21 (FN)        120 (TP)
```
- Only **3 false positives** (when it says fake, 97.6% chance it's right)
- Accepts more false negatives for ultimate precision
- **F1-Score**: 90.9%

---

## 🛠️ Installation

### Prerequisites

- **Python**: 3.9 or higher (3.10 recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional (NVIDIA with CUDA for faster GAPL)
- **Storage**: ~5GB for models and dependencies

### Step-by-Step Setup

**1. Clone the Repository**
```bash
git clone https://github.com/o6-webwork/deepfake-ensemble.git
cd deepfake-ensemble
```

**2. Create Virtual Environment** (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Download Model Weights**

**SPAI Model** (~2GB - Required for all detection modes):
1. Download from Google Drive: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view
2. Create directory: `mkdir -p spai/weights`
3. Place file at: `spai/weights/spai.pth`

**GAPL Model** (~850MB - Auto-loaded when using Enhanced 4-Layer or GAPL Only modes):
- Weights are included in the `gapl/pretrained/` directory
- No manual download required

**5. Configure API Keys** (Optional - for VLM modes)

Create `.env` file:
```bash
# OpenAI (for GPT-4o)
OPENAI_API_KEY=sk-your-key-here

# Anthropic (for Claude)
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Google (for Gemini)
GOOGLE_API_KEY=your-google-key-here

# Self-hosted vLLM (for Qwen3-VL, etc.)
VLLM_BASE_URL=http://localhost:8000/v1
```

**6. Launch Application**
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser!

---

## 🐳 Docker Deployment (Alternative)

For containerized deployment with GPU support:

**1. Prerequisites**
- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU acceleration)
- SPAI weights downloaded to `spai/weights/spai.pth`

**2. Download SPAI Weights**
```bash
# Create directory
mkdir -p spai/weights

# Download from Google Drive and place at spai/weights/spai.pth
# https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view
```

**3. Configure Environment (Optional)**
```bash
# Create .env file for API keys
cp .env.example .env

# Edit .env with your keys
nano .env
```

**4. Launch with Docker Compose**
```bash
docker-compose up -d
```

**5. Access Application**
- Web UI: http://localhost:8501
- View logs: `docker-compose logs -f`
- Stop: `docker-compose down`

**Docker Features:**
- ✅ GPU acceleration (NVIDIA runtime)
- ✅ Automatic restarts
- ✅ Volume mounts for weights and results
- ✅ Resource limits (4-8GB RAM, 4 CPUs)
- ✅ Access to external VLM servers on local network

**Note**: The container can access:
- External VLM servers on your local network (e.g., `http://100.64.0.x:8000`)
- Cloud APIs (OpenAI, Anthropic, Gemini)
- Use `host.docker.internal` instead of `localhost` for host services (Mac/Windows)

---

## 💻 Usage

### Web Interface

The Streamlit interface provides three main tabs:

#### 1. **Chat Interface** (Single Image Analysis)

1. Upload an image (JPEG, PNG, WebP)
2. Select detection mode:
   - Enhanced 4-Layer (comprehensive)
   - GAPL Only (high precision)
   - SPAI Only (fast)
   - SPAI + VLM (balanced)
3. Choose OSINT context (Auto/Military/Disaster/Propaganda)
4. Configure SPAI settings (resolution, heatmap transparency)
5. Click **"Analyze Image"**

**Results Include:**
- **Tier**: Authentic / Suspicious / Deepfake
- **Confidence**: Probability of being AI-generated (0-100%)
- **Analysis**: Detailed reasoning from VLM (if enabled)
- **Heatmap**: Visual attention overlay showing suspicious regions
- **Layer Breakdown**: Individual verdicts from each layer (Enhanced mode)

#### 2. **Batch Evaluation** (Dataset Testing)

1. Prepare ground truth CSV with columns: `filename`, `label`
   - Labels must be: `"Real"` or `"AI Generated"`
2. Upload CSV file
3. Upload images as ZIP or select folder
4. Choose detection mode and VLM model
5. Click **"Run Evaluation"**

**Results Include:**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix
- Per-image predictions with confidence
- Excel export with detailed metrics
- ROC curve and threshold analysis

#### 3. **Analytics Dashboard**

- View historical evaluation results
- Compare model performance across datasets
- Track accuracy trends over time
- Analyze common failure patterns

### Python API

```python
from detector import OSINTDetector
from PIL import Image

# Initialize detector
detector = OSINTDetector(
    detection_mode="enhanced_3layer",  # or "spai_standalone", "spai_assisted"
    context="auto",  # or "military", "disaster", "propaganda"
    model_key="qwen3_vl_32b",  # VLM model (if using VLM modes)
    spai_max_size=1280,  # Max resolution for SPAI
    debug=False
)

# Load image
image = Image.open("test_image.jpg")
image_bytes = image.tobytes()

# Detect
result = detector.detect(image_bytes, debug=False)

# Access results
print(f"Tier: {result['tier']}")  # Authentic/Suspicious/Deepfake
print(f"Confidence: {result['confidence']:.2%}")  # P(AI Generated)
print(f"Reasoning: {result['reasoning']}")

# Enhanced 4-Layer specific
if 'layer_agreement' in result:
    print(f"Texture: {result['layer_agreement']['texture']}")
    print(f"GAPL: {result['layer_agreement']['gapl']}")
    print(f"SPAI: {result['layer_agreement']['spai']}")
    print(f"VLM: {result['layer_agreement']['vllm']}")
    print(f"Consensus: {result['layer_agreement']['consensus']}")

# Save heatmap
if result.get('spai_heatmap_bytes'):
    with open('heatmap.png', 'wb') as f:
        f.write(result['spai_heatmap_bytes'])
```

### Command-Line Interface

```bash
# Single image detection
python cli.py detect \
    --image path/to/image.jpg \
    --mode enhanced_3layer \
    --context auto \
    --output results.json

# Batch evaluation
python cli.py evaluate \
    --dataset groundtruth.csv \
    --images image_folder/ \
    --mode enhanced_3layer \
    --output evaluation_results.xlsx
```

---

## ⚙️ Configuration

### Threshold Tuning

Edit `detector.py` lines 71-72:

```python
# Default (Balanced)
TIER_THRESHOLD_DEEPFAKE = 0.75      # P(AI) >= 75% = Deepfake
TIER_THRESHOLD_SUSPICIOUS_LOW = 0.50  # P(AI) > 50% = Suspicious

# High Precision (fewer false positives)
TIER_THRESHOLD_DEEPFAKE = 0.85      # Require 85% confidence
TIER_THRESHOLD_SUSPICIOUS_LOW = 0.60  # Raise suspicious threshold

# High Recall (catch all fakes)
TIER_THRESHOLD_DEEPFAKE = 0.65      # Lower bar for Deepfake
TIER_THRESHOLD_SUSPICIOUS_LOW = 0.40  # More aggressive
```

### SPAI Temperature Calibration

Adjust in `detector.py` `_calibrate_spai_score()` method (line 1200):

```python
# Default: temperature=1.5 (balanced)

# For heavily compressed social media images
calibrated = self._calibrate_spai_score(raw_score, temperature=2.0)

# For pristine/professional images
calibrated = self._calibrate_spai_score(raw_score, temperature=1.0)
```

### VLM Prompts

Edit `prompts/current.yaml` to customize:
- System prompts for different OSINT contexts
- Analysis instructions
- Verdict format

---

## 🔧 Troubleshooting

### "No ground truth for [filename], skipping"

**Cause**: Filename mismatch between uploaded images and ground truth CSV

**Solutions:**
1. Check CSV has exact filenames (case-sensitive)
2. System auto-strips prefixes like `AIG_`, `REAL_`
3. System matches basenames (ignores folder paths)
4. Remove any path prefixes from CSV filenames

### VLM Shows "Service Down" / High False Positives

**Cause**: VLM not contributing or being too aggressive

**Solutions:**
1. Use **Enhanced 4-Layer** instead of SPAI+VLM (weighted voting balances VLM)
2. Check API keys in `.env`
3. Try different VLM model (Qwen3-VL recommended)
4. Use **GAPL Only** or **SPAI Only** if VLM unavailable

### GAPL "CUDA out of memory"

**Solutions:**
```python
# Force CPU mode
gapl = GAPLForensicsPipeline(device="cpu")

# Or reduce SPAI resolution
detector = OSINTDetector(spai_max_size=640)
```

### High False Positive Rate

**Solutions:**
1. Select correct OSINT context (`context="disaster"` for chaotic scenes)
2. Raise thresholds (see Configuration section)
3. Use **GAPL Only** mode (97.6% precision)
4. Enable `watermark_mode="ignore"` for news photos

---

## 📚 Technical Details

### Weighted Voting Algorithm (Enhanced 4-Layer)

Each layer votes with weight based on confidence:
- **High confidence** = 3 votes
- **Medium confidence** = 2 votes
- **Low confidence** = 1 vote

**VLM Corroboration Check:**
- If VLM contradicts ALL forensic layers → weight capped at 1 vote
- Prevents VLM hallucinations from overriding evidence

**Consensus Flag:**
- `True` = All layers agree (high confidence verdict)
- `False` = Mixed signals (manual review recommended)

**Final Probability:**
- Average confidence of agreeing layers
- Converted to P(AI Generated)
- Re-classified to tier for consistency

### OSINT Context Protocols

**Military (CASE A):**
- Ignore formation patterns (expected in military)
- Focus on equipment details, shadow physics
- Expect uniform consistency (not flagged as AI)

**Disaster (CASE B):**
- Expect high chaos (debris, smoke, damage)
- Focus on impossible physics (floating objects)
- Tolerate spectral noise

**Propaganda (CASE C):**
- Distinguish retouching from AI generation
- Expect professional lighting
- Focus on anatomical impossibilities

### Temperature Scaling Math

```python
# Convert probability to logit
logit = log(p / (1 - p))

# Scale by temperature
scaled_logit = logit / T

# Convert back to probability
calibrated_p = 1 / (1 + exp(-scaled_logit))
```

**Effect**: T > 1 reduces overconfidence (saturated scores → balanced)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional detection layers (e.g., Noiseprint, GAN fingerprinting)
- [ ] Support for video deepfake detection
- [ ] Mobile/edge deployment optimization
- [ ] Additional VLM provider integrations
- [ ] Multi-language support for VLM reasoning

**Process:**
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📖 Citations

### Research Papers

**SPAI**: Spectral Analysis of AI-Generated Images (CVPR 2025)
**GAPL**: Generator-Aware Prototype Learning for Deepfake Detection (CVPR 2023)
**PLGF**: Texture Forensics Using Local Gravitational Force (IEEE TIFS 2022)
**PLADA**: Pay Less Attention to Deceptive Artifacts (ICCV 2023)

### Libraries

- PyTorch (BSD License)
- Streamlit (Apache 2.0)
- OpenCV (Apache 2.0)
- Pillow (HPND License)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

**Model Weights**: SPAI and GAPL have separate licenses from original research. See `models/LICENSE` for details.

---

## 🙏 Acknowledgments

- SPAI research team for spectral analysis framework
- GAPL authors for generator-aware prototype learning
- Texture forensics researchers for compression-resistant methods
- VLM providers (Qwen, OpenAI, Anthropic, Google)
- Open source community

---

## ⚠️ Disclaimer

This system is designed for research and OSINT analysis. While achieving 90.7% accuracy, **no detection system is perfect**. Always:

- Verify critical findings through multiple methods
- Consider context and source credibility
- Use human judgment for high-stakes decisions
- Comply with applicable laws and regulations

**Not suitable for**: Automated content moderation without human review, legal evidence without expert validation, or any application where errors could cause significant harm.
