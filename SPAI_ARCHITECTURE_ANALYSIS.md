# SPAI (Spectral AI-Generated Image Detector) - Comprehensive Architecture Analysis
## CVPR 2025 Implementation

---

## 1. OVERVIEW & KEY CONCEPTS

### Project Summary
SPAI is a CVPR2025 paper implementation for detecting AI-generated images using **spectral learning**. The model learns the spectral distribution of real images in a self-supervised manner and detects AI-generated images as out-of-distribution samples using spectral reconstruction similarity.

**Key Innovation**: Uses spectral analysis of images to identify artifacts in AI-generated content by comparing low-frequency and high-frequency components.

**Paper**: "Any-Resolution AI-Generated Image Detection by Spectral Learning"  
**Authors**: Dimitrios Karageorgiou, Symeon Papadopoulos, Ioannis Kompatsiaris, Efstratios Gavves

---

## 2. CORE ARCHITECTURE & MODELS

### 2.1 Model Types

SPAI implements **two main model types** for AI-generated image detection:

#### A. **MFViT (Masked Frequency Vision Transformer)** - Fixed Resolution
- **File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/models/sid.py` (lines 400-620)
- **Purpose**: Detects AI-generated images on fixed 224x224 resolution images
- **Input**: B x C x H x W (default 224x224)
- **Output**: Classification logit (1 for AI-generated, 0 for real)

**Architecture Components**:
```python
class MFViT(nn.Module):
    - vit: Vision Transformer (backbone)
    - features_processor: FrequencyRestorationEstimator (FRE)
    - cls_head: Classification head
    - frequencies_mask: Circular frequency mask (parameter)
    - backbone_norm: Normalization layer (ImageNet or CLIP)
```

**Data Flow**:
1. Input image normalized to [0, 1]
2. Split into low-frequency and high-frequency components via FFT
3. Three branches processed through ViT:
   - Original image → ViT → features
   - Low-frequency image → ViT → features
   - High-frequency image → ViT → features
4. Features passed through FrequencyRestorationEstimator
5. Classification head produces final prediction

#### B. **PatchBasedMFViT** - Arbitrary Resolution
- **File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/models/sid.py` (lines 39-350)
- **Purpose**: Detects AI-generated images on arbitrary resolution images
- **Input**: List of tensors with arbitrary H x W or single tensor with multiple resolutions
- **Output**: Classification logit per image

**Architecture Components**:
```python
class PatchBasedMFViT(nn.Module):
    - mfvit: Core MFViT model
    - img_patch_size: Fixed patch size (e.g., 224)
    - img_patch_stride: Stride between patches (overlapping patches)
    - patch_aggregator: Learnable parameter for cross-attention
    - Heads, scale, to_kv, to_out: Cross-attention components
```

**Key Difference**: 
- Divides arbitrary resolution images into overlapping patches
- Processes each patch independently through MFViT
- Aggregates patch predictions via learnable cross-attention mechanism
- Minimum patches requirement to prevent processing very small images

**Forward Methods**:
1. `forward_batch()`: Fixed 224x224 images
2. `forward_arbitrary_resolution_batch()`: Arbitrary resolution images without export
3. `forward_arbitrary_resolution_batch_with_export()`: Arbitrary resolution + heatmap export

#### C. **ClassificationVisionTransformer** - Simple VIT Classification
- **File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/models/sid.py` (lines 664-710)
- **Purpose**: Simple classification without spectral decomposition
- **Input**: B x C x H x W (arbitrary resolution with list support)
- **Output**: Classification logit

---

### 2.2 Vision Transformer Backbone Options

The model supports multiple backbone types via config:

**1. Standard ViT (Vision Transformer)**
- **File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/models/vision_transformer.py`
- **Default Configuration**:
  - Patch size: 16
  - Embedding dimension: 768
  - Depth (layers): 12
  - Number of heads: 12
  - Intermediate layers: All 12 layers extracted for multi-scale features
  - MLP ratio: 4

**2. CLIP Backbone**
- **File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/models/backbones.py` (lines 33-71)
- **Type**: CLIPBackbone class
- **Features**:
  - Uses pre-trained CLIP model (e.g., "ViT-B/16")
  - Frozen parameters (no gradient updates)
  - Extracts intermediate layer outputs via hooks
  - Uses CLIP-specific normalization (different from ImageNet)
  - Extracts features from `ln_2` (layer norm 2) outputs

```python
class CLIPBackbone(nn.Module):
    # Loads CLIP and registers hooks on ln_2 modules
    # Returns stacked intermediate layer outputs
    # Mean: (0.48145466, 0.4578275, 0.40821073)
    # Std: (0.26862954, 0.26130258, 0.27577711)
```

**3. DINOv2 Backbone**
- **File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/models/backbones.py` (lines 74-99)
- **Type**: DINOv2Backbone class
- **Features**:
  - Uses self-supervised pre-trained DINOv2 model
  - Supports multiple model sizes: dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
  - Configurable intermediate layers extraction
  - Returns stacked intermediate layer outputs

---

### 2.3 Spectral Filtering - Core Innovation

**File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/models/filters.py`

#### Key Functions:

**1. `filter_image_frequencies()`**
```python
def filter_image_frequencies(
    image: torch.Tensor,      # Input image B x C x H x W
    mask: torch.Tensor         # Frequency mask
) -> tuple[torch.Tensor, torch.Tensor]:  # (filtered_image, residual)
```
- Applies FFT (Fast Fourier Transform) to image
- Center-shifts FFT spectrum
- Multiplies by mask to extract specific frequencies
- Returns both filtered image and residual (inverse mask)
- Used to separate low-frequency and high-frequency components

**2. `generate_circular_mask()`**
```python
def generate_circular_mask(
    input_size: int,              # 224
    mask_radius_start: int,       # e.g., 16
    mask_radius_stop: Optional[int] = None,  # e.g., None
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:  # H x W binary mask
```
- Creates circular frequency mask
- Low-pass filter: keeps frequencies within radius_start
- Band-pass filter: keeps frequencies between radius_start and radius_stop
- Device-aware tensor creation

**3. `generate_centered_2d_coordinates_grid()`**
- Creates centered coordinate grid for frequency domain
- Used internally by circular mask generation

#### Frequency Decomposition Strategy:
```
Input Image (RGB)
    ↓
[FFT] → Frequency Spectrum (Complex)
    ↓
Circular Mask Application:
    ├─ Mask=1 (inside circle): LOW-FREQUENCY components (shapes, structure)
    └─ Mask=0 (outside circle): HIGH-FREQUENCY components (texture, noise)
    ↓
[IFFT] → Reconstructed Images
    ├─ Low-frequency image (smooth, basic structure)
    └─ High-frequency image (fine details, textures)
```

**Default Configuration** (from `config.py`):
- MASK_RADIUS1: 16 (for low-frequency extraction)
- MASK_RADIUS2: 999 (effectively no high-frequency cutoff)

---

### 2.4 Frequency Restoration Estimator (FRE)

**File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/models/sid.py` (lines 750-1100)

**Purpose**: Computes spectral reconstruction similarity features

```python
class FrequencyRestorationEstimator(nn.Module):
    def forward(
        self,
        x: torch.Tensor,              # Original features (B x N x L x D)
        low_freq: torch.Tensor,        # Low-freq features (B x N x L x D)
        hi_freq: torch.Tensor          # High-freq features (B x N x L x D)
    ) -> torch.Tensor:  # Output features (B x output_dim)
```

**Key Components**:

1. **Patch Projector** (optional):
   - Projects each spatial location across intermediate features
   - Dimensionality: B x N x L x D → projected features
   - Optional per-feature projection

2. **Similarity Computation** (Core Feature Extraction):
   ```
   For each intermediate layer:
       sim_x_low_freq = cosine_similarity(orig, low_freq)      # B x N x L
       sim_x_hi_freq = cosine_similarity(orig, hi_freq)        # B x N x L
       sim_low_hi = cosine_similarity(low_freq, hi_freq)       # B x N x L
       
   Compute statistics (mean, std) over spatial dimension:
       sim_x_low_mean, sim_x_low_std
       sim_x_hi_mean, sim_x_hi_std
       sim_low_hi_mean, sim_low_hi_std
   ```
   
   **Total features per layer**: 6 features (3 similarities × 2 statistics)

3. **Feature Aggregation**:
   - Concatenate features from all intermediate layers: 6 * N_layers
   - Example: 12 layers → 72 features
   - Optional: Include original features branch (projected)

4. **Output Feature Dimensions**:
   - Base: `6 * num_intermediate_layers`
   - With original branch: `+ projection_dim` (typically 1024)
   - Example: 6*12 + 1024 = 1096 dimensions

**Exportable Forward** (for ONNX):
- Replaces `torch.std()` with `exportable_std()` for ONNX compatibility
- Ensures standard deviation computation works in ONNX format

---

### 2.5 Classification Head

**File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/models/sid.py` (lines 1105-1120)

```python
class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,           # Features from FRE (e.g., 1096)
        num_classes: int,         # 1 for binary (sigmoid), 2+ for softmax
        mlp_ratio: int = 1,       # Expansion ratio for hidden layer
        dropout: float = 0.5
    ):
        # Sequential MLP:
        # Linear(input_dim → input_dim*mlp_ratio)
        #   ↓
        # ReLU activation
        #   ↓
        # Dropout(dropout)
        #   ↓
        # Linear(input_dim*mlp_ratio → input_dim*mlp_ratio)
        #   ↓
        # ReLU activation
        #   ↓
        # Dropout(dropout)
        #   ↓
        # Linear(input_dim*mlp_ratio → num_classes)
```

**Typical Configuration** (from `spai.yaml`):
- Input dimension: 1024-1096 (depends on features processor)
- MLP ratio: 3
- Dropout: 0.5
- Output: 1 for binary classification

**Note**: Uses 2-layer MLP with intermediate ReLU activations

---

## 3. INFERENCE PIPELINE

### 3.1 Entry Points & CLI Commands

**File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/__main__.py`

**Primary Inference Entry Point**:
```bash
python -m spai infer --input <input> --output <output> \
    --model <model_weights> --cfg <config>
```

**Command Implementation** (lines 482-562):
```python
@cli.command()
def infer(
    cfg: Path,                     # Config file
    batch_size: int = 1,          # Inference batch size
    input_paths: list[Path],      # Input images or CSVs
    input_csv_root_dir: list[Path],  # Root dirs for CSV relative paths
    split: str = "test",          # Data split to use
    lmdb_path: Optional[Path] = None,  # LMDB storage path
    model: Path = "./weights/spai.pth",  # Model weights
    output: Path = "./output",    # Output directory
    tag: str = "spai",            # Experiment tag
    resize_to: Optional[int] = None,  # Max resolution
    extra_options: tuple[str, str] = ()  # Config overrides
) -> None:
```

### 3.2 Model Loading & Initialization

**Step 1: Configuration Loading**
```python
from spai.config import get_config

config = get_config({
    "cfg": "configs/spai.yaml",
    "batch_size": 1,
    "test_csv": ["/path/to/images.csv"],
    "test_csv_root": ["/path/to/root"],
    "model": "weights/spai.pth",
    "output": "output",
    "opts": []  # Optional config overrides
})
```

**Step 2: Model Creation**
```python
from spai.models import build_cls_model

model = build_cls_model(config)  # Returns appropriate model based on config
model.cuda()

# Model selection logic (in build.py):
if config.MODEL.TYPE == "vit" and config.MODEL.SID_APPROACH == "single_extraction":
    model = build_cls_vit(config)       # ClassificationVisionTransformer
elif config.MODEL.TYPE == "vit" and config.MODEL.SID_APPROACH == "freq_restoration":
    model = build_mf_vit(config)        # MFViT or PatchBasedMFViT
```

**Step 3: Weights Loading**
```python
from spai.utils import load_pretrained

load_pretrained(
    config,
    model,
    logger,
    checkpoint_path="weights/spai.pth",
    verbose=False
)
```

### 3.3 Data Loading & Preprocessing

**File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/data/data_finetune.py`

**Dataset Class**: `CSVDataset`
```python
class CSVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: pathlib.Path,         # Path to CSV with image metadata
        csv_root_path: pathlib.Path,    # Root path for relative image paths
        split: str,                     # "train", "val", or "test"
        transform,                      # Albumentations transform
        path_column: str = "image",     # CSV column with image paths
        split_column: str = "split",    # CSV column with split info
        class_column: str = "class",    # CSV column with labels (0/1)
        views: int = 1,                 # Number of augmented views per image
        lmdb_storage: Optional[Path] = None  # Optional LMDB storage
    ):
```

**Data Loading Process**:
1. Parse CSV file (expects columns: image, split, class)
2. Filter rows by split (train/val/test)
3. Load image via FileSystemReader or LMDBFileStorageReader
4. Apply transforms (augmentation, normalization, conversion)
5. Return tuple: (augmented_images, label, dataset_index)

**Transform Pipeline** (via albumentations):
```python
def build_transform(is_train: bool, config):
    if is_train:
        # Augmentation transforms + normalization
        return A.Compose([
            A.Resize(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.GaussBlur(blur_limit=(3,9), p=0.5),
            A.GaussNoise(p=0.5),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    else:
        # Inference: minimal transforms
        return A.Compose([
            A.Resize(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
```

**Image Format Support**:
- JPEG, PNG, WEBP, BMP (via PIL)
- Stored in filesystem or LMDB archive
- Automatically converted to RGB 3-channel

### 3.4 Inference Forward Pass

**File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/__main__.py` (lines 1105-1202)

**Validation Function** (used for both validation and inference):
```python
@torch.no_grad()
def validate(
    config,
    data_loader,
    model,
    criterion,
    neptune_run,
    verbose: bool = True,
    return_predictions: bool = False  # Set to True for inference to get scores
):
```

**Inference Data Flow** for one batch:
```
Input Images (B x 3 x H x W) with values in [0, 1]
    ↓
Model Forward Pass:
    ├─ Frequency Decomposition (FFT)
    │   ├─ Low-frequency extraction
    │   └─ High-frequency extraction
    │
    ├─ Vision Transformer Features
    │   ├─ Original image → ViT → features (B x N_layers x L x D)
    │   ├─ Low-freq image → ViT → features
    │   └─ High-freq image → ViT → features
    │
    ├─ Frequency Restoration Estimator
    │   └─ Compute spectral similarities (6 per layer)
    │   → Output: B x feature_dim
    │
    ├─ Classification Head
    │   └─ MLP layers
    │   → Output: B x 1 (logits)
    │
    ├─ Sigmoid Activation
    │   → Output: B x 1 (probabilities in [0, 1])
    │
    └─ Predictions: scores for each image
        └─ 0.0-0.5: Real image
        └─ 0.5-1.0: AI-generated image
```

**Handling Multiple Image Views**:
```python
if images.size(dim=1) > 1:  # Multiple augmented views
    predictions = [model(images[:, i]) for i in range(images.size(dim=1))]
    predictions = torch.stack(predictions, dim=1)
    
    if config.TEST.VIEWS_REDUCTION_APPROACH == "max":
        output = predictions.max(dim=1).values
    elif config.TEST.VIEWS_REDUCTION_APPROACH == "mean":
        output = predictions.mean(dim=1)
```

### 3.5 Attention Heatmap Export

**Type**: `AttentionMask` (dataclass)
```python
@dataclasses.dataclass
class AttentionMask:
    mask: Optional[pathlib.Path] = None              # Attention mask image
    overlay: Optional[pathlib.Path] = None           # Overlay on original
    overlayed_image: Optional[pathlib.Path] = None   # Blended result
```

**Heatmap Generation** (in arbitrary resolution mode):
```python
# During forward pass with export_dirs provided:
output, attention_masks = model(
    images,
    feature_extraction_batch_size=config.MODEL.FEATURE_EXTRACTION_BATCH,
    export_dirs=[export_path_per_image]
)

# Generates visualization showing spectral inconsistencies
# Red/high areas indicate AI-generated artifacts
```

**File**: Function `save_image_with_attention_overlay()` in `spai/utils.py`

---

## 4. CONFIGURATION SYSTEM

**File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/config.py` (495 lines)

### 4.1 Default Configuration Structure

**Base Configuration** (initialized in `_C = CN()`):

#### DATA Section
```yaml
DATA:
  BATCH_SIZE: 128                 # Training batch size per GPU
  VAL_BATCH_SIZE: null           # Defaults to BATCH_SIZE if null
  TEST_BATCH_SIZE: null
  DATA_PATH: ""                  # Path to dataset CSV
  CSV_ROOT: ""                   # Root for relative paths in CSV
  TEST_DATA_PATH: []             # Test dataset CSVs (multiple)
  TEST_DATA_CSV_ROOT: []
  LMDB_PATH: null               # Optional LMDB storage
  DATASET: "csv_sid"            # Dataset type
  IMG_SIZE: 224                 # Input image size
  MIN_CROP_SCALE: 0.2
  INTERPOLATION: "bicubic"      # Resize interpolation
  PIN_MEMORY: true              # DataLoader pin memory
  NUM_WORKERS: 24               # Data loading workers
  PREFETCH_FACTOR: 2
  
  # Filtering parameters
  FILTER_TYPE: "mfm"            # "mfm", "sr", "deblur", "denoise"
  SAMPLE_RATIO: 0.5             # Low-pass filter sampling ratio
  MASK_RADIUS1: 16              # Low-frequency mask radius
  MASK_RADIUS2: 999             # High-frequency mask radius (effectively none)
  
  # Augmentation views
  AUGMENTED_VIEWS: 1            # Number of augmented views per sample
```

#### MODEL Section
```yaml
MODEL:
  TYPE: "vit"                   # Model architecture: "vit", "swin", "resnet"
  NAME: "finetune"
  REQUIRED_NORMALIZATION: "positive_0_1"  # Input normalization type
  RESOLUTION_MODE: "fixed"      # "fixed" or "arbitrary" resolution support
  FEATURE_EXTRACTION_BATCH: null  # Internal batch size for feature extraction
  
  # SID (Synthetic Image Detection) configuration
  SID_APPROACH: "freq_restoration"  # "single_extraction" or "freq_restoration"
  NUM_CLASSES: 2                # Binary classification
  DROP_RATE: 0.0
  SID_DROPOUT: 0.5             # Dropout in SID layers
  DROP_PATH_RATE: 0.1
  LABEL_SMOOTHING: 0.1
  
  # Vision Transformer backbone config
  VIT:
    PATCH_SIZE: 16
    IN_CHANS: 3
    EMBED_DIM: 768              # Token embedding dimension
    DEPTH: 12                   # Number of transformer blocks
    NUM_HEADS: 12               # Attention heads
    MLP_RATIO: 4                # MLP hidden dim ratio
    QKV_BIAS: true
    INIT_VALUES: 0.1            # Initialization scale
    USE_APE: true               # Absolute positional embedding
    USE_RPB: false              # Relative position bias
    USE_MEAN_POOLING: true
    USE_INTERMEDIATE_LAYERS: true  # Extract intermediate features
    INTERMEDIATE_LAYERS: [0,1,2,3,4,5,6,7,8,9,10,11]  # All 12 layers
    FEATURES_PROCESSOR: "rine"  # Feature processor type
    PROJECTION_DIM: 1024        # Projection dimension
    PROJECTION_LAYERS: 2        # Number of projection layers
    PATCH_PROJECTION: true      # Project each patch
    PATCH_PROJECTION_PER_FEATURE: true  # Per-feature projection
    PATCH_POOLING: "mean"       # "mean" or "l2_max"
    DECODER:
      EMBED_DIM: 512
      DEPTH: 0
      NUM_HEADS: 16
  
  # Frequency Restoration Estimator config
  FRE:
    MASKING_RADIUS: 16
    PROJECTOR_LAST_LAYER_ACTIVATION_TYPE: "gelu"  # or null for ReLU
    ORIGINAL_IMAGE_FEATURES_BRANCH: true  # Include original features
    DISABLE_RECONSTRUCTION_SIMILARITY: false
  
  # Classification head config
  CLS_HEAD:
    MLP_RATIO: 3                # Head MLP expansion ratio
  
  # Patch-based ViT for arbitrary resolution
  PATCH_VIT:
    PATCH_STRIDE: 224           # Stride between patches
    NUM_HEADS: 12               # Cross-attention heads
    ATTN_EMBED_DIM: 1536        # Cross-attention embedding dim
    MINIMUM_PATCHES: 1          # Minimum patches to process
```

#### TRAIN Section
```yaml
TRAIN:
  START_EPOCH: 0
  EPOCHS: 35                    # Total training epochs
  WARMUP_EPOCHS: 5
  BASE_LR: 5e-4                 # Learning rate
  WARMUP_LR: 2.5e-7
  MIN_LR: 2.5e-7
  WEIGHT_DECAY: 0.05
  LAYER_DECAY: 0.8              # Per-layer learning rate decay
  CLIP_GRAD: null              # Gradient clipping threshold
  MODE: "supervised"            # "supervised" or "contrastive"
  LOSS: "bce"                  # Loss function
  ACCUMULATION_STEPS: 1         # Gradient accumulation
  USE_CHECKPOINT: false         # Gradient checkpointing
  SCALE_LR: false              # Linear LR scaling with batch size
  
  OPTIMIZER:
    NAME: "adamw"              # AdamW optimizer
    EPS: 1e-8
    BETAS: (0.9, 0.999)
    MOMENTUM: 0.9              # For SGD
  
  LR_SCHEDULER:
    NAME: "cosine"             # Cosine annealing scheduler
    DECAY_EPOCHS: 30
    DECAY_RATE: 0.1
```

#### TEST Section
```yaml
TEST:
  CROP: true                    # Center crop during testing
  MAX_SIZE: null               # Max resolution (resize if larger)
  ORIGINAL_RESOLUTION: true     # Use original image resolution
  VIEWS_GENERATION_APPROACH: null  # "tencrop" or null
  VIEWS_REDUCTION_APPROACH: "mean"  # "mean" or "max" for multiple views
  EXPORT_IMAGE_PATCHES: false   # Export patch visualizations
  
  # Test-time perturbations
  GAUSSIAN_BLUR: false
  GAUSSIAN_NOISE: false
  JPEG_COMPRESSION: false
  SCALE: false
```

#### AUGMENTATION Section
```yaml
AUG:
  MIN_CROP_AREA: 0.2           # Minimum crop ratio
  HORIZONTAL_FLIP_PROB: 0.5
  VERTICAL_FLIP_PROB: 0.5
  ROTATION_PROB: 0.5
  ROTATION_DEGREES: 90
  GAUSSIAN_BLUR_PROB: 0.5
  GAUSSIAN_BLUR_LIMIT: (3, 9)
  GAUSSIAN_BLUR_SIGMA: (0.01, 0.5)
  GAUSSIAN_NOISE_PROB: 0.5
  JPEG_COMPRESSION_PROB: 0.5
  JPEG_MIN_QUALITY: 50
  JPEG_MAX_QUALITY: 100
  WEBP_COMPRESSION_PROB: 0.0
  COLOR_JITTER: 0.0
  AUTO_AUGMENT: "rand-m9-mstd0.5-inc1"  # RandAugment policy
  MIXUP: 0.8
  CUTMIX: 1.0
```

### 4.2 Default SPAI Configuration

**File**: `/home/otb-02/Desktop/deepfake detection/spai/configs/spai.yaml`

```yaml
MODEL:
  SID_APPROACH: "freq_restoration"
  TYPE: vit
  NAME: finetune
  DROP_PATH_RATE: 0.1
  NUM_CLASSES: 2
  REQUIRED_NORMALIZATION: "positive_0_1"
  RESOLUTION_MODE: "arbitrary"          # Arbitrary resolution support
  FEATURE_EXTRACTION_BATCH: 400
  VIT:
    EMBED_DIM: 768
    DEPTH: 12
    NUM_HEADS: 12
    INIT_VALUES: null
    USE_APE: True
    USE_RPB: False
    USE_SHARED_RPB: False
    USE_MEAN_POOLING: True
    USE_INTERMEDIATE_LAYERS: True
    PROJECTION_DIM: 1024
    PROJECTION_LAYERS: 2
    PATCH_PROJECTION: True
    PATCH_PROJECTION_PER_FEATURE: True
    INTERMEDIATE_LAYERS: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  FRE:
    MASKING_RADIUS: 16
    PROJECTOR_LAST_LAYER_ACTIVATION_TYPE: null
    ORIGINAL_IMAGE_FEATURES_BRANCH: True
  CLS_HEAD:
    MLP_RATIO: 3
  PATCH_VIT:
    MINIMUM_PATCHES: 4

DATA:
  DATASET: csv_sid
  IMG_SIZE: 224
  NUM_WORKERS: 8
  AUGMENTED_VIEWS: 4
  TEST_PREFETCH_FACTOR: 1

AUG:
  COLOR_JITTER: 0.

TRAIN:
  EPOCHS: 35
  WARMUP_EPOCHS: 5
  BASE_LR: 5e-4
  WARMUP_LR: 2.5e-7
  MIN_LR: 2.5e-7
  WEIGHT_DECAY: 0.05
  LAYER_DECAY: 0.8
  CLIP_GRAD: null
  LOSS: "bce"

TEST:
  ORIGINAL_RESOLUTION: True

PRINT_FREQ: 100
SAVE_FREQ: 10
```

### 4.3 Configuration Loading

```python
def get_config(args: dict) -> CfgNode:
    """Load configuration with argument overrides"""
    config = _C.clone()  # Clone default config
    update_config(config, args)  # Apply overrides
    return config

# Usage:
config = get_config({
    "cfg": "configs/spai.yaml",
    "batch_size": 1,
    "test_csv": ["/path/to/images.csv"],
    "output": "./output",
    "opts": [("MODEL.VIT.EMBED_DIM", "1024")]  # Runtime overrides
})
```

---

## 5. TRAINING PIPELINE

**File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/__main__.py`

### 5.1 Training Entry Point

```bash
python -m spai train --cfg configs/spai.yaml \
    --data-path datasets/images.csv \
    --csv-root-dir /data/root \
    --output output --tag experiment_name
```

### 5.2 Training Components

**Optimizer**: AdamW with layer decay
**Scheduler**: Cosine annealing with warmup
**Loss Function**: Binary cross-entropy (BCE) with optional supervised contrastive loss
**Mixed Precision**: NVIDIA Apex (O1, O2 options)

### 5.3 Training Loop

**Main Training Function** (lines 850-957):
```python
def train_model(
    config, model, model_without_ddp, data_loader_train, data_loader_val,
    data_loaders_test, dataset_val, datasets_test, datasets_test_names,
    criterion, optimizer, lr_scheduler, log_writer, neptune_run, save_all=False
):
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(...)  # Single epoch training
        acc, ap, auc, loss = validate(...)  # Validation
        # Save checkpoint if loss improved
        # Test on test sets
```

**Per-Epoch Training** (lines 960-1103):
```python
def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, ...):
    model.train()
    for batch in data_loader:
        # Handle multiple augmented views
        if num_views > 1:
            outputs_views = [model(samples[:, i]) for i in range(num_views)]
            outputs = torch.stack(outputs_views, dim=1)
        else:
            outputs = model(samples)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass with gradient accumulation
        if ACCUMULATION_STEPS > 1:
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            if (idx + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
        else:
            loss.backward()
            optimizer.step()
```

### 5.4 Model Checkpointing

**Saved Checkpoint Contents**:
```python
{
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'lr_scheduler': lr_scheduler.state_dict(),
    'max_accuracy': max_accuracy,
    'epoch': epoch,
    'config': config,
    'amp': amp.state_dict()  # If using mixed precision
}
```

---

## 6. UI/APPLICATION LAYER

**File**: `/home/otb-02/Desktop/deepfake detection/spai/app.py`

### 6.1 Streamlit Web Interface

**Purpose**: User-friendly web application for AI image detection

**Features**:
- Image upload (JPG, PNG, WEBP)
- Real-time analysis with progress bar
- Classification result display (real vs AI-generated)
- Attention heatmap visualization
- Configurable transparency blending
- Download detected results

### 6.2 Main Components

**Model Loading** (cached):
```python
@st.cache_resource
def load_model(device_name):
    config = get_config({"cfg": CONFIG_PATH, "batch_size": 1, "opts": []})
    model = build_cls_model(config)
    model.to(device_name)
    load_pretrained(config, model, logger, checkpoint_path=WEIGHTS_PATH)
    model.eval()
    return model, config
```

**User Interface**:
- **Sidebar Controls**:
  - Device selection (CUDA/CPU)
  - Max resolution slider (512 to 2048 or Original)
  - Attention heatmap checkbox
  - Overlay transparency slider

- **Main Content**:
  - Image upload widget
  - Original image display
  - Analysis progress bar
  - Results section with:
    - AI-Generated Probability (percentage)
    - Classification badge
    - Spectral Attention Map (if enabled)
    - Download button for blended image

### 6.3 Image Blending & Visualization

**Heatmap Creation**:
```python
def create_transparent_overlay(original_pil, overlay_path, alpha=0.6):
    # Load original image
    background = np.array(original_pil)
    
    # Load overlay/heatmap from file
    foreground = cv2.imread(str(overlay_path))
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    
    # Resize to match dimensions
    if foreground.shape[:2] != background.shape[:2]:
        foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))
    
    # Blend with controllable transparency
    blended = cv2.addWeighted(background, alpha, foreground, 1-alpha, 0)
    
    return blended
```

**Output**:
- Blended image showing spectral inconsistencies
- Red/high intensity areas indicate AI-generated artifacts
- Downloadable PNG/JPG result

---

## 7. DATA PROCESSING & TRANSFORMS

**File**: `/home/otb-02/Desktop/deepfake detection/spai/spai/data/data_finetune.py`

### 7.1 Transform Pipeline

**Inference Transform** (for integration):
```python
def build_transform(is_train: bool, config):
    if not is_train:  # Inference mode
        return A.Compose([
            A.Resize(
                height=config.DATA.IMG_SIZE,
                width=config.DATA.IMG_SIZE,
                interpolation=cv2.INTER_CUBIC
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()  # Converts to tensor, permutes to (C, H, W)
        ], keymap={"image": "image"})
```

### 7.2 Data Reader Abstraction

**Reader Interface**:
```python
class DataReader:
    def load_image(self, image_path: str, channels: int = 3) -> Image.Image:
        """Load image from path, return PIL Image"""
        pass
```

**Implementations**:
1. **FileSystemReader**: Loads from local filesystem
2. **LMDBFileStorageReader**: Loads from LMDB archive (faster for datasets)

### 7.3 CSV Dataset Format

**Expected CSV Columns**:
```
image,split,class
path/to/image1.jpg,train,0
path/to/image2.jpg,train,1
path/to/image3.jpg,val,0
path/to/image4.jpg,test,1
```

**Column Mapping** (configurable):
- `path_column="image"`: Column with image paths (relative to csv_root_path)
- `split_column="split"`: Split identifier (train/val/test)
- `class_column="class"`: Binary label (0=real, 1=AI-generated)

---

## 8. KEY FUNCTIONS & INTEGRATION POINTS

### 8.1 Core Model Loading

```python
# Path: spai/models/build.py
def build_cls_model(config) -> nn.Module:
    """Build classification model for inference"""
    model_type = config.MODEL.TYPE  # "vit"
    task_type = config.MODEL.SID_APPROACH  # "freq_restoration"
    
    if model_type == "vit" and task_type == "single_extraction":
        return build_cls_vit(config)
    elif model_type == "vit" and task_type == "freq_restoration":
        return build_mf_vit(config)
    else:
        raise NotImplementedError(f"Unknown cls model: {model_type}")
```

### 8.2 Simple Inference Function

For integration into existing deepfake detection system:

```python
from pathlib import Path
from spai.config import get_config
from spai.models import build_cls_model
from spai.utils import load_pretrained
from spai.data.data_finetune import build_transform
import torch
from PIL import Image
import numpy as np

class SPAIDetector:
    def __init__(
        self,
        config_path: str = "configs/spai.yaml",
        weights_path: str = "weights/spai.pth",
        device: str = "cuda"
    ):
        self.config = get_config({"cfg": config_path, "batch_size": 1, "opts": []})
        self.model = build_cls_model(self.config)
        self.model.to(device)
        load_pretrained(self.config, self.model, None, weights_path)
        self.model.eval()
        self.device = device
        self.transform = build_transform(is_train=False, config=self.config)
    
    def predict(self, image_path: str) -> dict:
        """
        Args:
            image_path: Path to image file
        
        Returns:
            {
                "score": float,  # 0-1, >0.5 = AI-generated
                "class": str,    # "real" or "ai_generated"
                "confidence": float  # Confidence level
            }
        """
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        
        # Apply transform
        input_tensor = self.transform(image=img_np)["image"]
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_batch)
            score = torch.sigmoid(output).item()
        
        return {
            "score": score,
            "class": "ai_generated" if score > 0.5 else "real",
            "confidence": max(score, 1 - score)
        }

# Usage:
detector = SPAIDetector()
result = detector.predict("test_image.jpg")
print(f"Image is {result['class']} with confidence {result['confidence']:.2%}")
```

### 8.3 Batch Inference

```python
class SPAIBatchDetector:
    def predict_batch(self, image_paths: list[str], batch_size: int = 4) -> list[dict]:
        """Process multiple images in batches"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                img_np = np.array(img)
                input_tensor = self.transform(image=img_np)["image"]
                batch_images.append(input_tensor)
            
            # Stack batch
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                scores = torch.sigmoid(outputs).cpu().numpy()
            
            # Parse results
            for path, score in zip(batch_paths, scores):
                results.append({
                    "image_path": path,
                    "score": float(score[0]),
                    "class": "ai_generated" if score[0] > 0.5 else "real",
                    "confidence": max(score[0], 1 - score[0])
                })
        
        return results
```

### 8.4 ONNX Export & Deployment

```bash
# Export to ONNX format for production deployment
python -m spai export-onnx \
    --cfg configs/spai.yaml \
    --model weights/spai.pth \
    --output onnx_models

# Validates ONNX model matches PyTorch
python -m spai validate-onnx \
    --cfg configs/spai.yaml \
    --model weights/spai.pth \
    --output onnx_models
```

---

## 9. DEPENDENCIES & REQUIREMENTS

**File**: `/home/otb-02/Desktop/deepfake detection/spai/requirements.txt`

Key dependencies:
- `torch>=2.0.0` - Deep learning framework
- `torchvision` - Vision utilities
- `timm` - PyTorch image models
- `albumentations` - Image augmentation
- `yacs` - Configuration management
- `click` - CLI framework
- `neptune-client` - Experiment tracking (optional)
- `opencv-python` - Image processing
- `Pillow` - Image library
- `scikit-image` - Image processing utilities
- `einops` - Tensor operations
- `clip` - CLIP model loading (optional)
- `torch-onnx` - ONNX export

**Hardware Requirements**:
- **Training**: 48GB GPU (e.g., Nvidia L40S)
- **Inference**: <8GB GPU RAM or CPU

---

## 10. KEY FILES REFERENCE

| Component | File Path | Lines | Description |
|-----------|-----------|-------|-------------|
| **Core Models** | `spai/models/sid.py` | 1270 | PatchBasedMFViT, MFViT, ClassificationVisionTransformer, FRE |
| **Backbones** | `spai/models/backbones.py` | 99 | CLIPBackbone, DINOv2Backbone |
| **Filters (FFT)** | `spai/models/filters.py` | 95 | Frequency filtering, mask generation |
| **VIT** | `spai/models/vision_transformer.py` | 667 | Vision Transformer architecture |
| **Model Builder** | `spai/models/build.py` | 22 | Factory functions |
| **Configuration** | `spai/config.py` | 495 | Config system, defaults |
| **CLI/Training** | `spai/__main__.py` | 1207 | Training, testing, inference, export |
| **Data Loading** | `spai/data/data_finetune.py` | 500+ | CSVDataset, transforms |
| **Inference** | `spai/__main__.py:482` | infer command | High-level inference |
| **Validation** | `spai/__main__.py:1106` | validate function | Metrics computation |
| **Streamlit UI** | `app.py` | 224 | Web interface |

---

## 11. INTEGRATION WITH DEEPFAKE DETECTION SYSTEM

### Quick Integration Example

```python
import sys
sys.path.insert(0, '/path/to/spai')

from spai.config import get_config
from spai.models import build_cls_model
from spai.utils import load_pretrained
from spai.data.data_finetune import build_transform
import torch
import numpy as np
from PIL import Image

# Initialize SPAI detector
config = get_config({
    "cfg": "spai/configs/spai.yaml",
    "batch_size": 1,
    "opts": []
})

model = build_cls_model(config)
model.cuda()
load_pretrained(config, model, None, "spai/weights/spai.pth")
model.eval()

transform = build_transform(is_train=False, config=config)

# Inference function
def detect_ai_generated(image_path: str) -> float:
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    
    input_tensor = transform(image=img_np)["image"]
    input_batch = input_tensor.unsqueeze(0).cuda()
    
    with torch.no_grad():
        output = model(input_batch)
        score = torch.sigmoid(output).cpu().item()
    
    return score  # >0.5 = AI-generated

# Usage in deepfake system
ai_gen_score = detect_ai_generated("test.jpg")
print(f"AI-Generated probability: {ai_gen_score:.2%}")
```

---

## 12. MODEL INFERENCE CONFIGURATION

**Recommended Settings for Integration**:

```python
{
    "MODEL": {
        "TYPE": "vit",
        "SID_APPROACH": "freq_restoration",
        "RESOLUTION_MODE": "arbitrary",  # Handles any resolution
        "FEATURE_EXTRACTION_BATCH": 400,
        "NUM_CLASSES": 2
    },
    "DATA": {
        "IMG_SIZE": 224,
        "BATCH_SIZE": 1  # Adjust based on GPU memory
    },
    "TEST": {
        "ORIGINAL_RESOLUTION": True,
        "VIEWS_REDUCTION_APPROACH": "mean"
    }
}
```

---

## SUMMARY

SPAI is a sophisticated AI-generated image detection model based on spectral learning. It combines:

1. **Spectral Decomposition**: FFT-based separation of images into low/high frequency components
2. **Vision Transformers**: ViT-B/16 backbone with intermediate layer extraction
3. **Frequency Restoration Estimator**: Computes spectral reconstruction similarity metrics
4. **Flexible Resolution**: Supports both fixed (224x224) and arbitrary resolution inputs
5. **Multiple Backbones**: Supports standard ViT, CLIP, and DINOv2
6. **Production Ready**: ONNX export, Streamlit UI, comprehensive CLI

Key for deepfake detection system integration:
- Fast inference (<100ms for 224x224 image)
- Arbitrary resolution support for varied input sizes
- Attention heatmaps for interpretability
- Well-documented API and configuration system
