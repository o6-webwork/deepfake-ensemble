#!/usr/bin/env python3
"""
NexInspect Initialization & Documentation Script

This script:
1. Validates project structure and dependencies
2. Generates comprehensive documentation
3. Creates quick reference files
4. Prevents token waste by caching project information

Run this script at the start of any session to get instant project context.

Usage:
    python init_project.py [--full] [--check-only]

Options:
    --full        Generate all documentation (default: quick summary only)
    --check-only  Only validate environment, don't generate docs
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class Colors:
    """Terminal colors for pretty output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}[OK] {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}[WARN] {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}[ERROR] {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}[INFO] {text}{Colors.ENDC}")

def check_project_structure() -> Tuple[bool, List[str]]:
    """
    Validate project structure and critical files.

    Returns:
        (is_valid, missing_files)
    """
    print_header("Validating Project Structure")

    critical_files = [
        "app.py",
        "detector.py",
        "spai_detector.py",
        "classifier.py",
        "config.py",
        "requirements.txt",
        "docker-compose.yml",
        "Dockerfile",
        "prompts/current.yaml",
    ]

    critical_dirs = [
        "spai",
        "spai/configs",
        "spai/weights",
        "prompts",
        "results",
        "analysis_output",
    ]

    missing_files = []
    missing_dirs = []

    # Check files
    for file in critical_files:
        if Path(file).exists():
            print_success(f"Found: {file}")
        else:
            print_error(f"Missing: {file}")
            missing_files.append(file)

    # Check directories
    for dir_path in critical_dirs:
        if Path(dir_path).exists():
            print_success(f"Found: {dir_path}/")
        else:
            print_warning(f"Missing: {dir_path}/")
            missing_dirs.append(dir_path)

    # Check SPAI weights (critical for inference)
    spai_weights = Path("spai/weights/spai.pth")
    if spai_weights.exists():
        size_mb = spai_weights.stat().st_size / (1024 * 1024)
        print_success(f"Found SPAI weights: {size_mb:.2f} MB")
    else:
        print_error("SPAI weights not found! Download from:")
        print_info("  https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view")
        print_info("  Place at: spai/weights/spai.pth")
        missing_files.append("spai/weights/spai.pth")

    is_valid = len(missing_files) == 0
    return is_valid, missing_files + missing_dirs

def check_model_configs() -> Dict:
    """
    Load and validate model configurations.

    Returns:
        Model configuration dictionary
    """
    print_header("Checking Model Configurations")

    # Check for models.json
    models_json = Path("models.json")
    if not models_json.exists():
        print_warning("models.json not found, checking for example...")
        example_json = Path("models.json.example")
        if example_json.exists():
            print_info("Found models.json.example - copy to models.json and configure")
            return {}
        else:
            print_error("No model configuration found!")
            return {}

    # Load models.json
    try:
        with open(models_json, 'r') as f:
            config = json.load(f)

        models = config.get("models", [])
        print_success(f"Loaded {len(models)} model configurations")

        # Group by provider
        providers = {}
        for model in models:
            provider = model.get("provider", "unknown")
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model.get("display_name", model.get("id", "unnamed")))

        print_info("\nConfigured providers:")
        for provider, model_names in providers.items():
            print(f"  • {provider.upper()}: {', '.join(model_names)}")

        return config

    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in models.json: {e}")
        return {}
    except Exception as e:
        print_error(f"Error loading models.json: {e}")
        return {}

def check_prompt_versions() -> Dict:
    """
    Check prompt version system.

    Returns:
        Prompt version information
    """
    print_header("Checking Prompt Versions")

    prompts_dir = Path("prompts")
    if not prompts_dir.exists():
        print_error("prompts/ directory not found!")
        return {}

    # Find all version files
    version_files = sorted(prompts_dir.glob("v*.yaml"))
    current_yaml = prompts_dir / "current.yaml"

    print_success(f"Found {len(version_files)} prompt versions:")
    for vfile in version_files:
        print(f"  • {vfile.name}")

    # Check current version
    if current_yaml.exists():
        try:
            with open(current_yaml, 'r') as f:
                current_prompts = yaml.safe_load(f)

            metadata = current_prompts.get('metadata', {})
            version = metadata.get('version', 'unknown')
            last_updated = metadata.get('last_updated', 'unknown')

            print_success(f"\nActive version: {version}")
            print_info(f"Last updated: {last_updated}")

            return {
                'version': version,
                'last_updated': last_updated,
                'available_versions': [vf.stem for vf in version_files]
            }
        except Exception as e:
            print_error(f"Error reading current.yaml: {e}")
            return {}
    else:
        print_error("prompts/current.yaml not found!")
        return {}

def check_docker_environment() -> bool:
    """
    Check Docker and nvidia-docker setup.

    Returns:
        True if Docker is available
    """
    print_header("Checking Docker Environment")

    # Check docker command
    docker_available = os.system("docker --version > /dev/null 2>&1") == 0
    if docker_available:
        print_success("Docker is available")
    else:
        print_warning("Docker not found (optional if running natively)")

    # Check nvidia-docker
    if docker_available:
        nvidia_docker = os.system("docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi > /dev/null 2>&1") == 0
        if nvidia_docker:
            print_success("NVIDIA Docker runtime available")
        else:
            print_warning("NVIDIA Docker runtime not available (GPU acceleration disabled)")

    return docker_available

def generate_quick_reference():
    """Generate QUICKREF.txt for instant project context"""
    print_header("Generating Quick Reference")

    quickref = f"""
==============================================================================
                    NEXINSPECT - QUICK REFERENCE
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==============================================================================

PROJECT STRUCTURE
------------------------------------------------------------------------------
Core Files:
  • app.py                    - Streamlit web interface (main entry point)
  • detector.py               - OSINT detection orchestrator
  • spai_detector.py          - SPAI spectral analysis wrapper
  • classifier.py             - VLM classification & verdict extraction
  • config.py                 - Model endpoint configurations
  • cloud_providers.py        - Multi-provider VLM adapters

Forensic Modules:
  • physics_forensics.py      - Eye reflection, lighting checks
  • texture_forensics.py      - PLGF, frequency, PLADA analysis
  • deepfake_detector.py      - 3-layer comprehensive detector

Utilities:
  • shared_functions.py       - Evaluation metrics, batch processing
  • generate_report_updated.py - Report generation
  • analytics.py              - Analytics dashboard
  • prompt_version.py         - Prompt version management

Configuration:
  • models.json               - VLM server endpoints (LOCAL ONLY - not in git)
  • prompts/current.yaml      - Active prompt version (symlink)
  • prompts/v*.yaml           - Versioned prompt files
  • docker-compose.yml        - Container orchestration
  • Dockerfile                - Application container definition

SPAI Model:
  • spai/configs/spai.yaml    - SPAI model configuration
  • spai/weights/spai.pth     - Pre-trained weights (~2GB, download separately)
  • spai/spai/                - SPAI source code (Vision Transformer)

DETECTION MODES
------------------------------------------------------------------------------
1. SPAI Standalone
   • Fast spectral analysis only
   • No VLM required
   • Best for: Batch processing, quick screening

2. SPAI + VLM Assisted
   • SPAI spectral + VLM semantic reasoning
   • Two-stage inference with KV-cache
   • Best for: OSINT analysis, comprehensive reports
   • Accuracy: 79.2% (with v1.2.0 prompts)

3. Enhanced 3-Layer
   • Physics + Texture + VLM analysis
   • Weighted voting system
   • Best for: High-stakes decisions, forensic verification

OSINT CONTEXTS
------------------------------------------------------------------------------
• Military:    Uniforms, parades, formations
• Disaster:    Floods, rubble, fires, BDA
• Propaganda:  Studio shots, news imagery, state media
• Auto:        Automatic context detection (default)

QUICK COMMANDS
------------------------------------------------------------------------------
Start Application:
  $ docker-compose up -d
  $ streamlit run app.py --server.port=8501 (native)

  Access UI: http://localhost:8501

Docker Management:
  $ ./deploy.sh start       # Build and start
  $ ./deploy.sh logs        # View logs
  $ ./deploy.sh status      # Check status
  $ ./deploy.sh test        # Test VLM connectivity
  $ ./deploy.sh stop        # Stop application
  $ ./deploy.sh clean       # Full cleanup

Prompt Version Management:
  $ python prompt_version.py info           # Show current version
  $ python prompt_version.py list           # List all versions
  $ python prompt_version.py activate 1.2.0 # Switch version
  $ python prompt_version.py diff 1.1.0 1.2.0  # Compare versions

Generate Reports:
  $ python generate_report_updated.py       # Analysis report
  $ python generate_analytics_report.py     # Analytics report

PERFORMANCE BENCHMARKS
------------------------------------------------------------------------------
SPAI Model Loading:
  • First load: One-time initialization per session
  • Subsequent: Cached
  • Memory: ~2GB VRAM

Inference Performance (GPU):
  • SPAI Standalone: Fast
  • SPAI + VLM: Moderate
  • Enhanced 3-Layer: Comprehensive (depends on forensic layers)

Hardware Requirements:
  • GPU: NVIDIA with 8GB+ VRAM (RTX A5000 tested)
  • RAM: 8GB minimum, 16GB recommended
  • Storage: 10GB (app + models + weights)

CONFIGURATION
------------------------------------------------------------------------------
VLM Providers:
  1. vLLM (Local):     InternVL, MiniCPM-V, Qwen3 VL
  2. OpenAI (Cloud):   GPT-4o, GPT-4o-mini
  3. Anthropic (Cloud): Claude 3.5 Sonnet, Haiku
  4. Gemini (Cloud):   Gemini 2.0 Flash, 1.5 Pro

Edit models.json to configure endpoints and API keys.

Environment Variables:
  • OPENAI_API_KEY      - OpenAI API key
  • ANTHROPIC_API_KEY   - Anthropic API key
  • GEMINI_API_KEY      - Google Gemini API key
  • CUDA_VISIBLE_DEVICES - GPU selection (default: 0)
  • STREAMLIT_SERVER_PORT - UI port (default: 8501)

PROMPT VERSIONS
------------------------------------------------------------------------------
Available Versions:
  • v1.0.0: Baseline (centralized verdict)        - 72% accuracy
  • v1.1.0: Bias mitigation (4-section format)    - 76% accuracy
  • v1.2.0: Tactical protocol (field-based) ★     - 79.2% accuracy [CURRENT BEST]

Switch versions with: python prompt_version.py activate <version>

TROUBLESHOOTING
------------------------------------------------------------------------------
SPAI Running on CPU (Slow):
  1. Check GPU access: docker exec deepfake-detector-app nvidia-smi
  2. Verify docker-compose.yml has GPU configuration
  3. Enable debug mode and check "SPAI Device" (should show cuda:0)

VLM Connection Errors:
  1. Test vLLM server: curl http://localhost:8000/v1/models
  2. Verify models.json endpoints match running servers
  3. Check API keys for cloud providers

Model Loading Fails:
  1. Verify SPAI weights exist: ls -lh spai/weights/spai.pth (should be ~2GB)
  2. Check Docker volume mount in docker-compose.yml
  3. Download weights from: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view

DOCUMENTATION FILES
------------------------------------------------------------------------------
• PIPELINE_OVERVIEW.md       - Complete system architecture (COMPREHENSIVE)
• README.md                  - Main project documentation
• QUICKSTART.md              - 5-minute setup guide
• README_DOCKER.md           - Docker deployment details
• ANALYTICS_README.md        - Analytics dashboard documentation
• EVALUATION_RESULTS.md      - Performance benchmarks
• DOCKER_SUCCESS.md          - Docker validation checklist
• QUICKREF.txt               - This file (generated by init_project.py)

USEFUL LINKS
------------------------------------------------------------------------------
• SPAI Paper:  https://github.com/HighwayWu/SPAI
• SPAI Weights: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view
• InternVL:    https://github.com/OpenGVLab/InternVL
• MiniCPM-V:   https://github.com/OpenBMB/MiniCPM-V
• vLLM Docs:   https://docs.vllm.ai/

------------------------------------------------------------------------------
For comprehensive details, see PIPELINE_OVERVIEW.md
------------------------------------------------------------------------------
"""

    with open("QUICKREF.txt", 'w', encoding='utf-8') as f:
        f.write(quickref)

    print_success("Generated QUICKREF.txt")
    print_info("View with: cat QUICKREF.txt")

def generate_file_map():
    """Generate comprehensive file map"""
    print_header("Generating File Map")

    file_map = """# NexInspect File Map

## Core Application Files

### Main Entry Points
- **app.py** (1,500+ lines)
  - Streamlit web interface
  - Tabs: Single Detection, Batch Evaluation, Analytics
  - Session state management
  - SPAI model caching (@st.cache_resource)
  - Interactive visualizations with Plotly

### Detection Pipeline
- **detector.py** (1,000+ lines)
  - OSINTDetector class: Main orchestrator
  - Detection modes: spai_standalone, spai_assisted, enhanced_3layer
  - SPAI integration and heatmap generation
  - VLM two-stage inference (analysis + verdict)
  - Metadata extraction and AI signature detection
  - Three-tier classification (Authentic/Suspicious/Deepfake)

- **spai_detector.py** (~500 lines)
  - SPAIDetector class: SPAI model wrapper
  - Spectral analysis preprocessing
  - Vision Transformer inference
  - Attention heatmap generation
  - Device management (CUDA/CPU)

- **deepfake_detector.py** (~800 lines)
  - EnhancedDeepfakeDetector class: 3-layer comprehensive analysis
  - Physics forensics layer integration
  - Texture analysis layer integration
  - VLM semantic layer integration
  - Weighted voting system
  - Confidence calibration

### VLM Integration
- **classifier.py** (~400 lines)
  - VLM classification logic
  - Verdict extraction with logprobs
  - MCQ-based scoring (A=Real, B=Fake)
  - Confidence calibration

- **cloud_providers.py** (~600 lines)
  - Multi-provider adapter pattern
  - Supported: vLLM, OpenAI, Anthropic, Gemini
  - Unified interface for all providers
  - API key management
  - Error handling and retries

### Forensic Analysis
- **physics_forensics.py** (~700 lines)
  - Eye reflection analysis (iris detection)
  - Lighting consistency checks (face vs. background)
  - Shadow validation
  - 3D geometry consistency

- **texture_forensics.py** (~1,200 lines)
  - PLGF (Phase-based Local Gradient Features)
  - Frequency domain analysis (DCT, DFT)
  - PLADA (Photo Response Non-Uniformity Analysis)
  - Artifact detection (blocking, ringing, color bleed)

### Utilities
- **shared_functions.py** (~500 lines)
  - analyze_single_image(): Single image analysis wrapper
  - chat_with_model(): VLM chat interface
  - load_ground_truth(): CSV loading for batch eval
  - calculate_metrics(): Accuracy, Precision, Recall, F1
  - display_confusion_matrix(): Plotly heatmap visualization

- **config.py** (~150 lines)
  - load_model_configs(): Load from models.json
  - get_default_model_configs(): Fallback hardcoded configs
  - MODEL_CONFIGS: Global model registry
  - SYSTEM_PROMPT: Base VLM system prompt
  - PROMPTS: Legacy prompt templates

### Analytics & Reporting
- **analytics.py** (~1,000 lines)
  - Analytics dashboard implementation
  - Performance comparison visualizations
  - Confusion matrix heatmaps
  - ROC curve analysis
  - Prediction viewer (filter by TP/TN/FP/FN)

- **generate_report_updated.py** (~800 lines)
  - Comprehensive report generation
  - Excel export (config + metrics + predictions sheets)
  - Timestamp and audit trail
  - Per-image result tracking

- **generate_analytics_report.py** (~600 lines)
  - PDF report generation with ReportLab
  - Embedded Plotly charts
  - Multi-configuration comparison
  - Best performer highlighting

### Prompt Management
- **prompt_version.py** (~400 lines)
  - info: Show current active version
  - list: List all available versions
  - activate: Switch to different version
  - bump: Create new version (major/minor/patch)
  - diff: Compare two versions
  - File-per-version system with symlinks

## Configuration Files

### Model Configuration
- **models.json** (user-created, not in git)
  - VLM server endpoints (local vLLM + cloud APIs)
  - API keys (local only, not committed)
  - Provider settings
  - Model display names

- **models.json.example**
  - Template for models.json
  - Example configurations for all providers
  - Copy to models.json and customize

### Prompts
- **prompts/current.yaml**
  - Symlink to active version file
  - Loaded by detector.py at initialization
  - Version metadata embedded

- **prompts/v1.0.0.yaml**
  - Baseline prompt version
  - Centralized verdict format
  - 72% accuracy baseline

- **prompts/v1.1.0.yaml**
  - Bias mitigation improvements
  - 4-section structured format
  - 76% accuracy

- **prompts/v1.2.0.yaml**
  - Tactical field-based protocol (CURRENT BEST)
  - 5-step analysis framework
  - 79.2% accuracy, F1 0.842

- **prompts.yaml** (legacy)
  - Old monolithic prompts file
  - Deprecated in favor of file-per-version system

### Docker Configuration
- **Dockerfile**
  - Python 3.9 slim base
  - PyTorch + CUDA support
  - SPAI dependencies
  - Streamlit installation
  - WORKDIR: /app

- **docker-compose.yml**
  - Service: deepfake-detector
  - Port: 8501 (Streamlit UI)
  - GPU configuration (nvidia runtime)
  - Volume mounts: weights, configs, results
  - Environment variables

- **.dockerignore**
  - Excludes: venv/, __pycache__, *.pyc
  - Reduces image size

- **.env** (local only, not in git)
  - API keys for cloud providers
  - Environment-specific settings

- **.env.example**
  - Template for .env file
  - Example variables

### Deployment
- **deploy.sh**
  - Deployment automation script
  - Commands: start, stop, restart, logs, status, test, clean
  - VLM connectivity testing
  - Container orchestration

- **requirements.txt**
  - Core Python dependencies
  - Streamlit, PyTorch, OpenCV, Pillow
  - Pandas, Plotly, ReportLab
  - OpenAI, Anthropic, Google AI clients

- **requirements-cloud.txt**
  - Cloud-specific dependencies
  - Lighter weight for cloud deployments

## SPAI Model Files

### Configuration
- **spai/configs/spai.yaml**
  - Model architecture parameters
  - Input resolution settings
  - Attention head configuration
  - Transformer depth and width

### Model Weights
- **spai/weights/spai.pth** (~2GB, download separately)
  - Pre-trained SPAI model weights
  - Download from: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view
  - Place in spai/weights/ directory

### SPAI Source Code
- **spai/spai/models/mfm.py**
  - Masked Feature Modeling Vision Transformer
  - SPAI core model architecture
  - Frequency-domain feature extraction

- **spai/spai/models/swin_transformer.py**
  - Swin Transformer backbone
  - Shifted window attention mechanism

- **spai/spai/models/vision_transformer.py**
  - Vision Transformer (ViT) base implementation
  - Patch embedding and positional encoding

- **spai/spai/models/backbones.py**
  - Model backbone factory
  - Supports multiple architectures

- **spai/spai/models/filters.py**
  - Frequency domain filters
  - DCT, DFT preprocessing

- **spai/spai/models/losses.py**
  - Training loss functions
  - Frequency-aware losses

- **spai/spai/data/data_mfm.py**
  - Masked Feature Modeling dataset
  - Preprocessing pipeline

- **spai/spai/data/readers.py**
  - Image reading and augmentation
  - Dataset loaders

- **spai/spai/config.py**
  - SPAI configuration utilities
  - YAML parsing

- **spai/spai/utils.py**
  - General utilities for SPAI
  - Device management, logging

## Output Directories

### Results
- **results/**
  - Batch evaluation outputs
  - Excel files with config + metrics + predictions
  - Timestamped filenames

### Analysis Output
- **analysis_output/**
  - Generated reports (Markdown, PDF)
  - Visualizations (PNG, HTML)
  - Dataset summaries
  - INDEX.txt: File index
  - README.md: Output directory documentation

### Testing Files
- **testing_files/**
  - Sample test images
  - Ground truth CSVs
  - Evaluation datasets

### Misc
- **misc/**
  - Miscellaneous data files
  - Temporary outputs

## Documentation Files

- **README.md** - Main project documentation
- **QUICKSTART.md** - 5-minute quick start guide
- **README_DOCKER.md** - Docker deployment guide
- **ANALYTICS_README.md** - Analytics dashboard documentation
- **EVALUATION_RESULTS.md** - Performance benchmarks
- **DOCKER_SUCCESS.md** - Docker validation checklist
- **GEMINI_FIX.md** - Gemini API integration notes
- **PIPELINE_OVERVIEW.md** - Complete architecture overview (MOST COMPREHENSIVE)
- **QUICKREF.txt** - Quick reference (generated by init_project.py)
- **FILE_MAP.md** - This file (generated by init_project.py)

## Git Files

- **.gitignore**
  - Excludes: venv/, __pycache__, *.pyc
  - Excludes: models.json, .env (local config)
  - Excludes: results/, analysis_output/ (generated outputs)
  - Excludes: spai/weights/ (large model weights)

## Key Architectural Patterns

### 1. Caching Strategy
```python
@st.cache_resource
def load_spai_detector():
    return SPAIDetector(...)  # Loaded once per session
```

### 2. Provider Adapter Pattern
```python
class CloudAdapter:
    def analyze_image(self, image, prompt, system_prompt):
        # Unified interface for all providers
```

### 3. Three-Tier Classification
```python
if metadata_flags:
    return "Deepfake", 1.0  # Auto-fail
elif spai_score > 0.8:
    return "Deepfake", spai_score
elif spai_score < 0.3:
    return "Authentic", 1 - spai_score
else:
    return "Suspicious", spai_score  # Uncertain
```

### 4. Two-Stage VLM Inference
```python
# Stage 1: Analysis (creates KV-cache)
analysis = vlm.analyze(image, context_prompt)

# Stage 2: Verdict (reuses KV-cache)
verdict = vlm.extract_verdict(analysis, mcq_prompt)
```

### 5. Weighted Voting (3-Layer)
```python
final_score = (
    physics_score * PHYSICS_WEIGHT +
    texture_score * TEXTURE_WEIGHT +
    vlm_score * VLM_WEIGHT
) / (PHYSICS_WEIGHT + TEXTURE_WEIGHT + VLM_WEIGHT)
```

## Import Dependencies

### Core Dependencies
- streamlit - Web UI framework
- torch, torchvision - PyTorch for SPAI
- opencv-python (cv2) - Image processing
- PIL (Pillow) - Image I/O
- pandas - Data handling
- numpy - Numerical operations
- plotly - Interactive charts
- reportlab - PDF generation

### API Clients
- openai - OpenAI + vLLM client
- anthropic - Anthropic Claude client
- google.generativeai - Gemini client

### Utilities
- yaml - YAML parsing
- json - JSON configuration
- pathlib - Path handling
- datetime - Timestamps
- base64 - Image encoding for VLM APIs

## File Size Estimates

- **app.py**: 60KB (1,500 lines)
- **detector.py**: 45KB (1,000 lines)
- **spai_detector.py**: 20KB (500 lines)
- **deepfake_detector.py**: 30KB (800 lines)
- **physics_forensics.py**: 25KB (700 lines)
- **texture_forensics.py**: 50KB (1,200 lines)
- **spai/weights/spai.pth**: 2GB (model weights)
- **Total codebase (excluding weights)**: ~10MB

---

**Last Updated**: 2025-12-19
**Generated by**: init_project.py
"""

    with open("FILE_MAP.md", 'w', encoding='utf-8') as f:
        f.write(file_map)

    print_success("Generated FILE_MAP.md")
    print_info("View with: cat FILE_MAP.md or open in editor")

def main():
    """Main initialization routine"""
    import argparse

    parser = argparse.ArgumentParser(description="NexInspect Project Initialization")
    parser.add_argument("--full", action="store_true", help="Generate all documentation")
    parser.add_argument("--check-only", action="store_true", help="Only validate environment")
    args = parser.parse_args()

    print("\n")
    print(f"{Colors.BOLD}{Colors.OKCYAN}")
    print("=" * 70)
    print("                                                                   ")
    print("              NexInspect Project Initializer                      ")
    print("                                                                   ")
    print("          Advanced Deepfake Detection System                      ")
    print("          SPAI + Vision-Language Models                           ")
    print("                                                                   ")
    print("=" * 70)
    print(f"{Colors.ENDC}\n")

    # Phase 1: Structure validation
    is_valid, missing_items = check_project_structure()

    # Phase 2: Configuration checks
    model_config = check_model_configs()
    prompt_info = check_prompt_versions()
    docker_available = check_docker_environment()

    if args.check_only:
        print_header("Validation Complete")
        if is_valid:
            print_success("All critical files present")
        else:
            print_error(f"Missing {len(missing_items)} items:")
            for item in missing_items:
                print(f"  • {item}")
        return

    # Phase 3: Documentation generation
    generate_quick_reference()

    if args.full:
        generate_file_map()
        print_success("Full documentation generated")

    # Final summary
    print_header("Initialization Summary")

    print(f"{Colors.BOLD}Project Status:{Colors.ENDC}")
    if is_valid:
        print_success("All critical components present")
    else:
        print_error(f"Missing {len(missing_items)} items (see above)")

    print(f"\n{Colors.BOLD}Quick Start:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}1. Review QUICKREF.txt for instant reference{Colors.ENDC}")
    print(f"{Colors.OKCYAN}2. Review PIPELINE_OVERVIEW.md for complete architecture{Colors.ENDC}")
    print(f"{Colors.OKCYAN}3. Configure models.json with your VLM endpoints{Colors.ENDC}")
    print(f"{Colors.OKCYAN}4. Start application: docker-compose up -d{Colors.ENDC}")
    print(f"{Colors.OKCYAN}5. Access UI: http://localhost:8501{Colors.ENDC}")

    if not is_valid:
        print(f"\n{Colors.WARNING}{Colors.BOLD}Action Required:{Colors.ENDC}")
        for item in missing_items:
            print(f"{Colors.WARNING}  - {item}{Colors.ENDC}")

    print(f"\n{Colors.BOLD}Documentation Files:{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  [OK] QUICKREF.txt        - Quick reference guide{Colors.ENDC}")
    if args.full:
        print(f"{Colors.OKGREEN}  [OK] FILE_MAP.md         - Comprehensive file map{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  [OK] PIPELINE_OVERVIEW.md - Complete architecture (pre-existing){Colors.ENDC}")

    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  $ cat QUICKREF.txt                    # View quick reference{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  $ python prompt_version.py info       # Check prompt version{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  $ ./deploy.sh test                    # Test VLM connectivity{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  $ docker-compose up -d                 # Start application{Colors.ENDC}")

    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}Initialization complete!{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

if __name__ == "__main__":
    main()
