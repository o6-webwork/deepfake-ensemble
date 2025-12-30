# GAPL Model Weights

This directory should contain the GAPL checkpoint file.

## Download Required

**File**: `checkpoint.pt` (1.2 GB)

**Download from HuggingFace**:
```bash
wget https://huggingface.co/AbyssLumine/GAPL/resolve/main/checkpoint.pt
```

Or visit: https://huggingface.co/AbyssLumine/GAPL

**Place at**: `gapl/pretrained/checkpoint.pt`

## Verification

After downloading, verify the file exists:
```bash
ls -lh gapl/pretrained/checkpoint.pt
# Should show ~1.2GB file
```

The GAPL model will not work without this checkpoint file.
