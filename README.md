# NexInspect

Advanced deepfake detection system combining SPAI (Spectral AI-Generated Image Detector) with Vision-Language Models for comprehensive image authenticity analysis.

## 🎯 Overview

This system provides two detection modes optimized for OSINT analysis of military, disaster, and propaganda imagery:

- **SPAI Standalone**: Fast spectral analysis (~5s) using frequency-domain deep learning
- **SPAI + VLM**: Comprehensive analysis combining SPAI with semantic reasoning from Vision-Language Models

## ✨ Key Features

### Detection Capabilities
- **Dual-mode detection**: Choose between speed (SPAI only) or comprehensiveness (SPAI + VLM)
- **OSINT context awareness**: Specialized protocols for military, disaster, and propaganda scenarios
- **Visual explanations**: Attention heatmaps showing suspicious regions with warm color highlighting
- **Batch evaluation**: Process multiple images with configurable parameters and audit trail
- **Analytics dashboard**: Interactive visualization and PDF reporting for comparing model/prompt performance

### Technical Features
- **GPU-accelerated inference**: ~5 seconds per image on NVIDIA GPUs
- **Model caching**: Instant subsequent inferences after first load
- **Docker deployment**: Full containerization with GPU support
- **Multiple VLM providers**: Support for vLLM, OpenAI, Anthropic, and Google Gemini
- **Prompt version control**: File-per-version system with easy rollback and comparison
- **Excel reporting**: Comprehensive evaluation exports with config, metrics, and per-image results

## 🚀 Quick Start

### Prerequisites

- Docker with NVIDIA GPU support ([nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- SPAI model weights ([download link](https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view))
- VLM server endpoint (local or cloud)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/o6-webwork/deepfake-detection.git
   cd deepfake-detection
   ```

2. **Download SPAI weights**
   ```bash
   # Place the downloaded spai.pth file in:
   mkdir -p spai/weights
   # Copy spai.pth to spai/weights/spai.pth
   ```

3. **Configure VLM endpoints**

   Copy and edit the model configuration:
   ```bash
   cp models.json.example models.json
   # Edit models.json with your VLM endpoints
   ```

4. **Start the application**
   ```bash
   docker-compose up -d
   ```

5. **Access the web interface**

   Open your browser to: http://localhost:8501

## 📖 Usage

### Single Image Analysis

1. Upload an image via the web interface
2. Select detection mode:
   - **SPAI Standalone**: Fast spectral analysis only
   - **SPAI + VLM**: Comprehensive analysis with semantic reasoning
3. Configure SPAI parameters (Advanced Settings):
   - Resolution: 512-2048 or Original (default: 1280)
   - Heatmap transparency: 0.0-1.0 (default: 0.6)
4. Click "Analyze Image"

**Output includes:**
- Classification tier (Authentic/Suspicious/Deepfake)
- Confidence score (0-100%)
- Detailed reasoning
- Blended attention heatmap (warm colors = suspicious regions)

### Batch Evaluation

1. Navigate to the "Batch Evaluation" tab
2. Upload multiple images
3. Upload ground truth CSV with columns: `filename`, `label`
4. Configure evaluation settings:
   - OSINT context (auto/military/disaster/propaganda)
   - Detection mode (SPAI standalone or SPAI + VLM)
   - SPAI resolution
5. Select VLM models to evaluate (if using SPAI + VLM)
6. Click "Run Evaluation"
7. Download Excel report with results

**Excel Export contains:**
- **config**: Evaluation parameters for audit trail
- **metrics**: Accuracy, precision, recall, F1, confusion matrix per model
- **predictions**: Per-image results with analysis text

### Analytics Dashboard

1. Navigate to the "Analytics" tab
2. Upload evaluation results (Excel files from Batch Evaluation)
3. Explore interactive visualizations:
   - **Overview & Metrics**: Comparative performance charts, confusion matrices, ROC analysis
   - **Prediction Viewer**: Filter and inspect individual predictions by TP/TN/FP/FN
4. Export comprehensive PDF reports with embedded charts

**Features:**
- **Multi-configuration comparison**: Compare different model/prompt versions side-by-side
- **Interactive filtering**: Filter predictions by outcome type (TP/TN/FP/FN)
- **Best performers**: Automatically highlights top configurations by F1 score
- **PDF export**: Professional reports with embedded Plotly charts for offline review
- **Performance metrics**: Accuracy, precision, recall, F1, confusion matrices, and more

**Example workflow:**
```bash
# 1. Run batch evaluation (creates Excel file)
# 2. Upload Excel to Analytics tab
# 3. Compare prompt versions (e.g., v1.1.0 vs v1.2.0)
# 4. Export PDF report for documentation
```

See [ANALYTICS_README.md](ANALYTICS_README.md) for detailed analytics documentation.

## 🎛️ Configuration

### VLM Model Configuration

Edit `models.json` to configure your VLM endpoints:

```json
{
  "InternVL 2.5 8B": {
    "provider": "vllm",
    "base_url": "http://100.64.0.1:8000/v1",
    "model_name": "OpenGVLab/InternVL2_5-8B",
    "max_tokens": 4096
  },
  "GPT-4o": {
    "provider": "openai",
    "api_key": "your-api-key",
    "model_name": "gpt-4o",
    "max_tokens": 4096
  }
}
```

**Supported providers:**
- `vllm`: Local vLLM servers
- `openai`: OpenAI API (GPT-4V, GPT-4o)
- `anthropic`: Anthropic API (Claude 3)
- `gemini`: Google Gemini API

### OSINT Context Protocols

The system includes specialized detection protocols for different scenarios:

- **Military**: Uniforms, parades, formations - distinguishes natural patterns from AI duplication
- **Disaster**: Floods, earthquakes, fires - handles chaotic scenes appropriately
- **Propaganda**: Studio shots, news imagery - differentiates retouching from generation
- **Auto**: Automatically applies all protocols

### Prompt Version Control

The system uses a file-per-version approach for managing analysis prompts:

```bash
# Show current active version
python prompt_version.py info

# List all available versions
python prompt_version.py list

# Switch to a different version
python prompt_version.py activate 1.0.0

# Create new version
python prompt_version.py bump minor

# Compare two versions
python prompt_version.py diff 1.0.0 1.2.0
```

**Available versions:**
- **v1.0.0**: Baseline with centralized verdict prompt
- **v1.1.0**: Bias mitigation with structured 4-section report format
- **v1.2.0**: Simplified tactical protocol (current - best performance)
  - 79.2% accuracy, 83.3% recall, F1 0.842
  - Field-based output: SCENE, HIGH_RISK_ARTIFACTS, LOGIC_FAILURES, STYLISTIC_FLAGS, SPAI_HOTSPOTS
  - 5-step analysis: Scene Context, Human/Biological, Physics/Logic, Dramatic Tropes, Spectral Correlation

All versions are immutable and stored in `prompts/` directory. The active version is `prompts/current.yaml`.

### Docker Configuration

GPU access is configured in `docker-compose.yml`:

```yaml
services:
  deepfake-detector:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Environment variables:**
- `CUDA_VISIBLE_DEVICES`: Select GPU device (default: 0)
- `STREAMLIT_SERVER_PORT`: Web interface port (default: 8501)

## 📊 Performance

### SPAI Standalone Mode
- **First inference**: ~2 minutes (model load) + 5 seconds (inference)
- **Subsequent inferences**: ~5 seconds (GPU cached)
- **No VLM required**: Perfect for batch processing

### SPAI + VLM Mode
- **Total time**: ~8 seconds (5s SPAI + 3s VLM)
- **Comprehensive analysis**: Spectral + semantic reasoning
- **Context-aware**: OSINT protocols guide VLM attention

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (tested on RTX A5000)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for application + models

## 🔬 How It Works

### SPAI Spectral Analysis

SPAI (Spectral AI-Generated Image Detector) is a CVPR 2025 Vision Transformer that analyzes frequency-domain patterns to detect AI-generated content:

1. **Preprocessing**: Image converted to spectral representation
2. **Feature extraction**: Masked Feature Modeling ViT processes frequency patterns
3. **Classification**: Binary prediction (Real/AI-Generated) with confidence score
4. **Attention visualization**: Heatmap highlights suspicious regions

**Color interpretation:**
- **Dark red**: Highest confidence of AI manipulation
- **Orange/Yellow**: Moderate suspicion
- **Light to dark blue**: Low confidence (likely authentic)

### VLM Integration (SPAI + VLM Mode)

When SPAI + VLM mode is selected:

1. **SPAI analysis**: Spectral analysis generates score + heatmap
2. **Context selection**: OSINT protocol selected based on image type
3. **VLM reasoning**: Model analyzes image + heatmap with context-specific guidelines
4. **Verdict extraction**: Final classification via logprob-based scoring

The VLM receives:
- Original image
- SPAI attention heatmap overlay
- SPAI classification and confidence
- Context-specific detection protocols
- Metadata analysis results

## 🛠️ Development

### Project Structure

```
.
├── app.py                      # Streamlit web interface
├── detector.py                 # Main detection pipeline
├── spai_detector.py           # SPAI model wrapper
├── classifier.py              # VLM classification logic
├── shared_functions.py        # Evaluation utilities
├── prompts.yaml               # Detection prompts and protocols
├── config.py                  # Application configuration
├── models.json.example        # VLM endpoint template
├── docker-compose.yml         # Container orchestration
├── Dockerfile                 # Application container
├── requirements.txt           # Python dependencies
└── spai/                      # SPAI model source code
    ├── configs/
    ├── spai/                  # Core SPAI implementation
    └── weights/               # Model weights (download separately)
```

### Running Without Docker

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up SPAI weights**
   ```bash
   # Download and place in spai/weights/spai.pth
   ```

3. **Run the application**
   ```bash
   streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   ```

### Debug Mode

Enable debug mode in the UI to see:
- Performance timing breakdown
- SPAI raw scores and predictions
- VLM request latency
- EXIF metadata analysis
- Raw logprobs and top-k predictions

## 📝 VLM Server Setup

### Local vLLM Server

For best performance, run VLM servers locally:

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model OpenGVLab/InternVL2_5-8B \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

Update `models.json`:
```json
{
  "InternVL 2.5 8B": {
    "provider": "vllm",
    "base_url": "http://localhost:8000/v1",
    "model_name": "OpenGVLab/InternVL2_5-8B"
  }
}
```

### Cloud Providers

The system supports cloud VLM providers:

**OpenAI:**
```json
{
  "GPT-4o": {
    "provider": "openai",
    "api_key": "sk-...",
    "model_name": "gpt-4o"
  }
}
```

**Anthropic:**
```json
{
  "Claude 3.5 Sonnet": {
    "provider": "anthropic",
    "api_key": "sk-ant-...",
    "model_name": "claude-3-5-sonnet-20241022"
  }
}
```

**Google Gemini:**
```json
{
  "Gemini 2.0 Flash": {
    "provider": "gemini",
    "api_key": "...",
    "model_name": "gemini-2.0-flash-exp"
  }
}
```

## 🔍 Troubleshooting

### SPAI Running on CPU (Slow)

If inference takes 60+ seconds instead of 5 seconds:

1. **Verify GPU access in container:**
   ```bash
   docker exec deepfake-detector-app nvidia-smi
   ```

2. **Check Docker GPU configuration:**
   Ensure `docker-compose.yml` has GPU resources configured

3. **Enable debug mode** and check "SPAI Device" field:
   - Should show: `cuda:0`
   - If shows `cpu`, GPU access is not working

### VLM Connection Errors

1. **Check VLM server is running:**
   ```bash
   curl http://localhost:8000/v1/models
   ```

2. **Verify endpoint in models.json** matches server address

3. **For cloud providers:** Check API key is valid

### Model Loading Fails

1. **Verify SPAI weights exist:**
   ```bash
   ls -lh spai/weights/spai.pth
   # Should be ~2GB
   ```

2. **Check Docker volume mount:**
   ```yaml
   volumes:
     - ./spai/weights:/app/spai/weights:ro
   ```

## 📄 License

This project integrates SPAI under its original license. See `spai/LICENSE` for details.

## 🙏 Acknowledgments

- **SPAI**: Spectral AI-Generated Image Detector ([CVPR 2025](https://github.com/HighwayWu/SPAI))
- **Vision-Language Models**: Various providers (OpenAI, Anthropic, vLLM community)
- **Streamlit**: Web interface framework

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation in the repository

---

**Note**: This system is designed for OSINT analysis and research purposes. Always verify critical findings through multiple methods.
