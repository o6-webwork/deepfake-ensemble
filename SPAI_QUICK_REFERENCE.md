# SPAI - Quick Reference Guide

## Project Location
- **Full Path**: `/home/otb-02/Desktop/deepfake detection/spai/`
- **Main Module**: `spai/`
- **Config File**: `configs/spai.yaml`
- **Weights**: `weights/spai.pth` (needs to be downloaded)

## Key File Locations

### Core Models
| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **MFViT + PatchBasedMFViT** | `spai/models/sid.py` | 1270 | Main detection models |
| **Frequency Filtering** | `spai/models/filters.py` | 95 | FFT-based spectral decomposition |
| **Classification Head** | `spai/models/sid.py` | 1105-1120 | Final prediction layer |
| **VIT Backbone** | `spai/models/vision_transformer.py` | 667 | Transformer architecture |
| **Alternative Backbones** | `spai/models/backbones.py` | 99 | CLIP, DINOv2 support |

### Data & Config
| Component | File | Purpose |
|-----------|------|---------|
| **Configuration** | `spai/config.py` | Settings and defaults |
| **Dataset Class** | `spai/data/data_finetune.py` | CSV-based dataset loading |
| **Data Module Init** | `spai/data/__init__.py` | Data loader factory |

### Entry Points
| Function | File | Lines | Purpose |
|----------|------|-------|---------|
| **Training** | `spai/__main__.py` | train command | Model training |
| **Testing** | `spai/__main__.py` | test command | Model evaluation |
| **Inference** | `spai/__main__.py` | infer command | Predict on new images |
| **ONNX Export** | `spai/__main__.py` | export-onnx command | Model deployment |

### UI
| Component | File | Purpose |
|-----------|------|---------|
| **Streamlit App** | `app.py` | Web interface |

---

## Quick Start - Inference

### 1. Setup
```bash
cd /home/otb-02/Desktop/deepfake\ detection/spai
conda create -n spai python=3.11
conda activate spai
pip install -r requirements.txt
```

### 2. Download Weights
- Get from: Google Drive link (see README.md)
- Place in: `weights/spai.pth`

### 3. Inference
```bash
# Single directory of images
python -m spai infer --input /path/to/images --output output/

# CSV with image paths
python -m spai infer --input images.csv --output output/
```

### 4. Expected Output
- CSV file: `output/spai_epoch_X.csv`
- Columns: image path, AI-generated score (0-1)
- Scores >0.5 = AI-generated, <0.5 = Real

---

## Core Architecture Overview

### Model Types
1. **MFViT** (Fixed Resolution)
   - Input: 224x224 images
   - Best for: Consistent resolution datasets

2. **PatchBasedMFViT** (Arbitrary Resolution)
   - Input: Any resolution (dividable into 224x224 patches)
   - Best for: Mixed resolution datasets
   - Default in `spai.yaml`

3. **ClassificationVisionTransformer** (Simple VIT)
   - Input: Arbitrary resolution
   - No spectral analysis (alternative approach)

### Key Innovation: Spectral Analysis
```
Input Image → FFT → Frequency Mask Application
    ↓
Low-Frequency Branch    High-Frequency Branch    Original Branch
    ↓                           ↓                        ↓
Vision Transformer → Extract Intermediate Features
    ↓
Compute Spectral Similarities (6 per layer × 12 layers = 72 features)
    ↓
Classification Head (MLP) → Prediction (0-1 scale)
```

### Spectral Similarity Features
For each layer, compute:
1. Similarity(Original, Low-Freq): mean, std
2. Similarity(Original, High-Freq): mean, std  
3. Similarity(Low-Freq, High-Freq): mean, std
**Total**: 6 statistics per layer

---

## Configuration

### Default SPAI Config (`configs/spai.yaml`)
```yaml
MODEL:
  SID_APPROACH: "freq_restoration"  # Use spectral analysis
  RESOLUTION_MODE: "arbitrary"      # Arbitrary resolution support
  FEATURE_EXTRACTION_BATCH: 400     # Internal batch size
  VIT:
    EMBED_DIM: 768
    DEPTH: 12
    NUM_HEADS: 12
    INTERMEDIATE_LAYERS: [0,1,2,3,4,5,6,7,8,9,10,11]  # All layers
  FRE:
    MASKING_RADIUS: 16              # Cutoff for low-frequency mask
    ORIGINAL_IMAGE_FEATURES_BRANCH: True
  CLS_HEAD:
    MLP_RATIO: 3

DATA:
  IMG_SIZE: 224
  AUGMENTED_VIEWS: 4               # Training augmentation

TEST:
  ORIGINAL_RESOLUTION: True         # Use original size during inference
```

### Runtime Config Override
```python
config = get_config({
    "cfg": "configs/spai.yaml",
    "batch_size": 1,
    "test_csv": ["images.csv"],
    "output": "results",
    "opts": [("MODEL.VIT.EMBED_DIM", "1024")]  # Override
})
```

---

## Integration into Existing System

### Minimal Integration (3 steps)
```python
from pathlib import Path
from spai.config import get_config
from spai.models import build_cls_model
from spai.utils import load_pretrained
from spai.data.data_finetune import build_transform
import torch
from PIL import Image
import numpy as np

# 1. Initialize
config = get_config({"cfg": "configs/spai.yaml", "batch_size": 1, "opts": []})
model = build_cls_model(config)
model.cuda()
load_pretrained(config, model, None, "weights/spai.pth")
model.eval()
transform = build_transform(is_train=False, config=config)

# 2. Load image
img = Image.open("test.jpg").convert("RGB")
input_tensor = transform(image=np.array(img))["image"].unsqueeze(0).cuda()

# 3. Predict
with torch.no_grad():
    score = torch.sigmoid(model(input_tensor)).item()
    label = "AI-Generated" if score > 0.5 else "Real"
```

---

## Expected Performance

### Training Time
- **GPU**: ~2-3 hours per epoch (48GB L40S)
- **Epochs**: 35 (can stop earlier via early stopping)
- **Total**: ~100 hours

### Inference Speed
- **Per Image (224x224)**: ~50-100ms (GPU)
- **Per Image (arbitrary res)**: ~200-500ms (GPU, depends on size)
- **CPU**: ~10x slower

### Model Size
- **Weights**: ~345MB (ViT-B/16)
- **ONNX Export**: ~350MB
- **Memory (inference)**: <2GB (batch size 1)

---

## Data Format

### CSV Format
```csv
image,split,class
images/real_001.jpg,train,0
images/ai_001.jpg,train,1
images/real_test_001.jpg,test,0
images/ai_test_001.jpg,test,1
```

**Columns**:
- `image`: Relative path to image (relative to CSV root)
- `split`: One of "train", "val", "test"
- `class`: 0=real, 1=AI-generated

### CSV Root
- If not specified: parent directory of CSV file
- If specified: all relative paths resolved from this directory

---

## Important Parameters

### For Inference
| Parameter | Value | Effect |
|-----------|-------|--------|
| `FEATURE_EXTRACTION_BATCH` | 400 | Internal batch size for patch processing |
| `RESOLUTION_MODE` | "arbitrary" | Handle any image size |
| `ORIGINAL_RESOLUTION` | true | Use original image size (no resize to 224) |
| `MAX_SIZE` | null | No limit on input size |

### For Training
| Parameter | Value | Effect |
|-----------|-------|--------|
| `BASE_LR` | 5e-4 | Learning rate |
| `EPOCHS` | 35 | Total training epochs |
| `WEIGHT_DECAY` | 0.05 | L2 regularization |
| `LAYER_DECAY` | 0.8 | Per-layer LR scaling |
| `AUGMENTED_VIEWS` | 4 | Augmentation diversity |

---

## Common Issues & Solutions

### Issue: Model weights not found
```
FileNotFoundError: weights/spai.pth
```
**Solution**: Download weights from Google Drive link and place in `weights/` directory

### Issue: Out of memory (GPU)
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Reduce `batch_size` (use 1 for inference)
- Set `FEATURE_EXTRACTION_BATCH` to smaller value (e.g., 100)
- Use CPU instead (slower but less memory)

### Issue: Image format not supported
```
UnidentifiedImageError: cannot identify image file
```
**Solution**: Convert to PNG, JPG, or WEBP using PIL or ImageMagick

### Issue: CSV paths relative to wrong directory
**Solution**: Use `--input-csv-root-dir` to specify correct root path:
```bash
python -m spai infer --input images.csv --input-csv-root-dir /data/root
```

---

## Supported Backbones

### 1. Vision Transformer (Default)
- Pre-trained on ImageNet or MFM
- Most tested configuration
- ~345MB weights

### 2. CLIP
```python
# In config: set MODEL_WEIGHTS = "clip"
```
- Vision-language pre-training
- Different normalization
- Frozen backbone only

### 3. DINOv2
```python
# In config: set MODEL_WEIGHTS = "dinov2"
```
- Self-supervised learning
- Multiple sizes available (vitb14, vitl14, vitg14)
- Requires download on first use

---

## Exportable Formats

### ONNX Export (for production)
```bash
python -m spai export-onnx --cfg configs/spai.yaml \
    --model weights/spai.pth --output onnx_models/

# Output files:
# - onnx_models/patch_encoder.onnx
# - onnx_models/patch_aggregator.onnx
```

**Benefits**:
- Cross-platform compatibility
- Hardware acceleration (TensorRT, CoreML, etc.)
- Smaller file sizes
- Reproducible inference

---

## References

**Paper**: "Any-Resolution AI-Generated Image Detection by Spectral Learning"  
**Authors**: Karageorgiou et al., CVPR 2025  
**GitHub**: Official SPAI repository  

**Key Concepts**:
- Spectral analysis using FFT
- Vision Transformers for feature extraction
- Multi-scale intermediate layer features
- Spectral reconstruction similarity metrics

---

## File Size Reference

| File | Size | Purpose |
|------|------|---------|
| `weights/spai.pth` | ~345MB | Model weights |
| `onnx_models/patch_encoder.onnx` | ~175MB | Encoder ONNX |
| `onnx_models/patch_aggregator.onnx` | ~175MB | Aggregator ONNX |
| SPAI codebase | ~50MB | Source code |

---

## Contact & Support

**Implementation**: Centre for Research and Technology Hellas (CERTH), University of Amsterdam  
**License**: Apache 2.0  
**Repository**: Include link to official GitHub  

