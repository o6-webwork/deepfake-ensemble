# Add batch evaluation, reporting, and Docker containerization

## Summary

This PR introduces major enhancements to transform the deepfake detection application from a simple single-file script into a comprehensive evaluation platform with Docker support.

### Key Features Added

#### 🔄 Batch Evaluation System
- Refactored into modular Streamlit application with dual-tab interface
- **Tab 1**: Single Image Detection with interactive chat
- **Tab 2**: Batch Evaluation with automated metrics calculation
- Support for 4 VLM models: InternVL 2.5/3.5 8B, MiniCPM-V 4.5, Qwen3 VL 32B
- Consensus voting mechanism (5 runs per image, majority vote)
- Automated confusion matrix calculation and Excel export

#### 📊 Performance Analysis & Reporting
- Automated report generation with comprehensive metrics
- 7 PNG visualizations ready for presentation decks
- Markdown reports (Executive Summary + Detailed Analysis)
- Multi-domain breakdown: HADR, Military Conflict, Military Showcase
- AIG vs AIM performance analysis for binary classification
- Scenario-level tracking (12 scenarios: FIRE, FLOOD, EARTHQUAKE, etc.)

#### 🐳 Docker Containerization
- Complete Docker setup with optimized Dockerfile (Python 3.11-slim)
- Docker Compose configuration with resource limits
- Volume mounts for results, test data, and analysis outputs
- Helper deployment script with Docker Compose v2 support
- Comprehensive deployment documentation

#### 📚 Design Documentation
- Logit calibration conceptual guide addressing "Timidity Bias"
- Updated design spec for forensic image scanner with ELA/FFT analysis
- Research into VLM log-probability extraction

### Evaluation Results

- **Dataset**: 72 images (24 Real, 24 AIG, 24 AIM) across 3 domains
- **Best Model**: Qwen3 VL 32B achieving 55.56% accuracy
- **Evaluation Runs**: November 26 and December 2, 2025

### Technical Details

**Files Changed**: 33 files, 4,610 insertions
- Core application: `app.py`, `config.py`, `shared_functions.py`
- Reporting: `generate_report_updated.py`, visualizations in `analysis_output/`
- Docker: `Dockerfile`, `docker-compose.yml`, `deploy.sh`
- Documentation: `README_DOCKER.md`, `QUICKSTART.md`

### Bug Fixes
- Fixed domain extraction logic for 3-category dataset
- Corrected scenario parsing (FIRE vs LIVEFIRE substring issue)
- Resolved Docker apt-get package installation errors
- Updated for Docker Compose v2 compatibility

## Test Plan

- [x] Streamlit app runs successfully in local environment
- [x] Docker container builds and runs on http://localhost:8501
- [x] Batch evaluation generates Excel files with correct metrics
- [x] Report generation creates all 7 visualizations
- [x] All 4 VLM models connect successfully via vLLM endpoints

## Screenshots

The PR includes 7 visualization outputs:
1. Model Comparison (Overall Performance)
2. Confusion Matrices (All Models)
3. Dataset Composition
4. Performance Metrics Table
5. Detection Rates by Category
6. Confusion Matrix Definitions
7. Domain-Level Performance

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
