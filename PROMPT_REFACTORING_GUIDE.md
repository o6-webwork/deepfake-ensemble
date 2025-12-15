# Prompt Refactoring Guide

## Overview

This document describes the YAML-based prompt configuration system that replaces hardcoded prompts in `detector.py`.

## Changes Summary

### 1. New Files

**prompts.yaml**
- Centralized configuration for all prompt templates
- Contains system prompts, analysis instructions, and watermark handling
- Easy to edit without touching Python code

### 2. Modified Files

**detector.py**
- Added YAML loading via `_load_prompts()` classmethod
- System prompts now loaded from YAML templates
- Analysis instructions use comprehensive 73-point checklist
- Extended `max_tokens` from 1000 → 2000

**requirements.txt**
- Added `pyyaml>=6.0.0` dependency

**Dockerfile**
- Added `COPY prompts.yaml .` to include configuration in container

## YAML Structure

```yaml
system_prompts:
  base: "Main system prompt..."
  simplified: "Simplified version (no forensics)..."
  forensic_protocols:
    case_a: "Military context protocol..."
    case_b: "Disaster context protocol..."
    case_c: "Propaganda context protocol..."

analysis_instructions:
  visual_analysis: "73-point detailed checklist..."
  forensic_instructions: "ELA/FFT interpretation guide..."
  metadata_instructions: "Metadata analysis instructions..."

watermark_instructions:
  analyze: "Active watermark detection..."
  ignore: "DO NOT consider watermarks..."
```

## Enhanced Visual Analysis

The new system includes a comprehensive 6-category inspection framework:

### Primary Analysis Dimensions
1. **Scene realism** - Physical world vs 3D/surreal
2. **Object defects & anomalies** - Shape, texture, perspective errors
3. **Lighting & shadows** - Consistency and directionality
4. **Focus & depth of field** - Blur appropriateness
5. **Sharpness consistency** - Resolution and noise distribution
6. **Object interactions** - Occlusion and contact physics
7. **AI texture artifacts** - Brush strokes, over-smoothing
8. **Stylistic clues** - AI-characteristic styles

### Detailed Artifact Checklists

**1. Geometric and Structural Consistency**
- Perspective & Lighting (shadow conflicts, depth distortion)
- Physical Details (cloth folding, glass refraction)
- Biological Structures (finger count, ear symmetry, limb accuracy)
- Text Accuracy (signboard coherence, multi-line text)
- Edges & Seams (hair blending, area continuity)
- Smudging in Complex Areas (crowd/leaf simplification)

**2. Semantic and Common Sense Consistency**
- Scene Logic (impossible natural phenomena)
- Fantasy/Unreal Elements (magical or absurd objects)
- Over-Idealization (excessive symmetry, perfection)
- Repeated Textures (looping patterns)
- Uniform Micro-Expressions (identical facial expressions in groups)
- Abnormal Object Interactions (physical law violations)

**3. Indoor Scene Artifacts**
- Structural Integrity (wall/door/window connectivity)
- Spatial Logic (floating objects, wall clipping)
- Mirror Reflections (perspective consistency)
- Materials & Textures (stretching, seams)
- Perspective Consistency (vanishing point alignment)
- Semantic Consistency (furniture proportions, layout logic)
- Lighting & Shadows (source consistency)

**4. Human-Related Artifacts**
- Eyes (size/color/highlight asymmetry)
- Teeth (edge definition, smoothness)
- Ears/Accessories (size deviations, earring mismatch)
- Hair (texture distortion, gravity violations)
- Hands/Body (finger deformities, feature uniformity)
- Background Characters (missing details, object shape errors)

**5. Outdoor Scene Artifacts**
- Structural Integrity (building/road completeness)
- Spatial Logic (floating, sinking, depth reversal)
- Occlusion Relationships (layer ordering)
- Materials & Textures (pattern repetition, stitching)
- Perspective Consistency (single vanishing point)
- Semantic Consistency (scale, realistic combinations)
- Lighting & Shadows (unified direction/intensity)

**6. Target Object Artifacts**
- Symmetry (eye sizes, tire shapes)
- Edge Transition (boundary clarity)
- Icons/Text (license plates, labels)
- Structural Logic (shape consistency)
- Component Integrity (missing parts)
- Shadows & Reflections (consistency with lighting)
- Object Interactions (ground contact accuracy)
- Unreal Objects (absurd structures)
- Background Issues (perspective errors)

## Usage

### For Developers

The detector automatically loads prompts from `prompts.yaml` during initialization:

```python
detector = OSINTDetector(
    base_url="http://localhost:8000/v1",
    model_name="model-name",
    prompts_path="prompts.yaml"  # Optional, defaults to prompts.yaml
)
```

### For Prompt Engineers

Edit `prompts.yaml` directly to:
- Refine analysis instructions
- Adjust forensic protocols
- Modify system role descriptions
- Update watermark handling behavior

Changes take effect on next detector initialization (app restart).

### In Docker

The YAML file is baked into the container during build. To update prompts:
1. Edit `prompts.yaml` locally
2. Rebuild container: `docker compose up --build -d`

## Testing

Verified behaviors:
- ✅ YAML file loads successfully
- ✅ System prompts (full + simplified) generate correctly
- ✅ Forensic protocols (CASE A/B/C) conditionally included
- ✅ Visual analysis checklist properly integrated
- ✅ Watermark instructions dynamically selected
- ✅ All existing logic preserved (toggles, modes, contexts)

## Benefits

1. **Maintainability**: Prompts separated from code logic
2. **Iteration Speed**: No Python changes needed for prompt tweaks
3. **Comprehensiveness**: 73-point forensic-grade checklist
4. **Scalability**: Easy to add new categories or protocols
5. **Version Control**: Prompt changes clearly visible in git diffs
6. **Collaboration**: Non-developers can improve prompts

## Migration Notes

### What Changed
- Hardcoded prompt strings moved to YAML
- Analysis instructions vastly expanded (4 bullet points → 73 inspection points)
- max_tokens increased 1000 → 2000 to handle detailed output

### What Stayed the Same
- Forensic toggle behavior (send_forensics parameter)
- Watermark mode switching (ignore vs analyze)
- Context-specific protocols (military/disaster/propaganda)
- CASE A/B/C conditional logic
- Temperature settings (still 0.0 for determinism)

### Backward Compatibility
- All existing function signatures preserved
- Default behavior unchanged
- Docker deployment workflow identical
- API responses format unchanged

## Future Improvements

Potential enhancements:
- Multi-language prompt support
- A/B testing different prompt versions
- Dynamic prompt selection based on image type
- User-uploadable custom prompts via UI
- Prompt template variables for parameterization
