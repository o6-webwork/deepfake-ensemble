# Evaluation System Migration Plan: Score-Based → A/B Verdict

## Overview

This document outlines the plan to migrate the batch evaluation system from the old score-based verdict format (0-10 scale) to the new A/B logprob-based verdict system while maintaining backward compatibility with existing results data.

## Current System Analysis

### Old System (Score-Based)
**File:** `shared_functions.py::analyze_single_image()`
- **Input:** PIL Image
- **Output:** `{"classification": str, "analysis": str, "score": float}`
- **Verdict Logic:**
  - Extracts score from text using regex: `r"(?:score|final score)\s*:?\s*\**\s*(\d+(?:\.\d+)?)"`
  - Classification: `"AI Generated"` if score > 4, else `"Real"`
  - Score range: 0-10 (or similar numeric scale)
- **Storage:** Excel files with `score_example` column containing numeric values (0-9)

### New System (A/B Logprob-Based)
**File:** `detector.py::OSINTDetector`
- **Input:** `image_bytes` (bytes)
- **Output:** `{"tier": str, "confidence": float, "reasoning": str, ...}`
- **Verdict Logic:**
  - Stage 2: VLM analysis with comprehensive reasoning
  - Stage 3: Binary MCQ with logprobs extraction
    - Prompt: "(A) Real (Authentic Capture) / (B) Fake (AI Generated/Manipulated)"
    - Parses logprobs for tokens: `REAL_TOKENS = ['A', ' A', 'a', ' a']` and `FAKE_TOKENS = ['B', ' B', 'b', ' b']`
  - Softmax normalization → `confidence_fake` (0.0-1.0)
  - Three-tier classification:
    - `confidence_fake >= 0.90` → "Deepfake"
    - `0.50 <= confidence_fake < 0.90` → "Suspicious"
    - `confidence_fake < 0.50` → "Authentic"
- **Token Extraction:** Uses `max_tokens=1` with `logprobs=True` and `top_logprobs=5`
- **Confidence:** Derived from softmax of logprobs, not text-based extraction

## Key Differences

| Aspect | Old System | New System |
|--------|------------|------------|
| **Verdict Source** | Text parsing (regex) | Logprobs (probabilistic) |
| **Output Format** | Numeric score (0-10) | Token (A/B) + confidence (0-1) |
| **Classification** | Binary (AI Generated / Real) | Three-tier (Deepfake / Suspicious / Authentic) |
| **Confidence** | Score magnitude | Softmax probability |
| **Forensics** | Not integrated | Optional ELA/FFT artifacts |
| **Prompt Strategy** | Single-stage with score request | Two-stage (analysis + verdict) |
| **Context Awareness** | Generic | OSINT-specific (military/disaster/propaganda) |

## Migration Strategy

### Phase 1: Backward Compatibility Layer

**Goal:** Allow evaluation system to work with both old score-based data and new A/B verdict data.

**Implementation:**
1. **Add Format Detection** in `shared_functions.py`:
   ```python
   def detect_verdict_format(data: dict) -> str:
       """Detect whether data is score-based or logprob-based."""
       if 'score' in data and isinstance(data['score'], (int, float)):
           return 'score'
       elif 'confidence' in data and 'token' in data:
           return 'logprob'
       else:
           return 'unknown'
   ```

2. **Create Normalization Function**:
   ```python
   def normalize_verdict(data: dict, format_type: str) -> dict:
       """
       Normalize verdict to standard format regardless of source.

       Returns:
           {
               'classification': str,  # "AI Generated" or "Real"
               'confidence': float,   # 0.0-1.0
               'tier': str,          # "Authentic", "Suspicious", or "Deepfake"
               'raw_data': dict      # Original data for reference
           }
       """
       if format_type == 'score':
           # Old system: score 0-10
           score = data['score']
           classification = "AI Generated" if score > 4 else "Real"
           # Map score to confidence (0-10 → 0.0-1.0)
           confidence = score / 10.0
           # Map to three-tier (approximate)
           if score >= 9:
               tier = "Deepfake"
           elif score >= 5:
               tier = "Suspicious"
           else:
               tier = "Authentic"

       elif format_type == 'logprob':
           # New system: A/B token + confidence
           token = data['token']
           confidence_fake = data['confidence']
           classification = "Real" if token in ['A', ' A', 'a', ' a'] else "AI Generated"
           confidence = confidence_fake if classification == "AI Generated" else (1.0 - confidence_fake)
           # Use detector's three-tier logic
           if confidence_fake >= 0.90:
               tier = "Deepfake"
           elif confidence_fake >= 0.50:
               tier = "Suspicious"
           else:
               tier = "Authentic"

       return {
           'classification': classification,
           'confidence': confidence,
           'tier': tier,
           'raw_data': data
       }
   ```

### Phase 2: Update Evaluation Functions

**Goal:** Modify batch evaluation to use new detector system while storing data in compatible format.

**Files to Modify:**
1. **`shared_functions.py`**:
   - Keep `analyze_single_image()` for backward compatibility
   - Add new function `analyze_single_image_v2()` using `OSINTDetector`
   - Add `extract_verdict_from_detector()` to convert detector output to evaluation format

2. **`app.py` (Batch Evaluation Tab)**:
   - Add toggle: "Use Legacy Scoring" vs "Use A/B Verdict System"
   - When A/B selected:
     - Initialize `OSINTDetector` instead of using `analyze_single_image()`
     - Call `detector.detect(image_bytes, debug=False, send_forensics=True/False)`
     - Extract verdict from detector output
     - Store in Excel with new columns: `verdict_token`, `confidence_fake`, `tier`
   - Maintain old columns for compatibility: `score_example` (computed from confidence)

3. **Excel Schema Update**:
   ```
   OLD columns:
   - score_example: int (0-10)
   - classification: str ("AI Generated" / "Real")

   NEW columns (additional):
   - verdict_token: str ("A" / "B" / null for old data)
   - confidence_fake: float (0.0-1.0)
   - tier: str ("Authentic" / "Suspicious" / "Deepfake")
   - score_example: int (computed as int(confidence_fake * 10) for compatibility)
   ```

### Phase 3: Update Report Generator

**Goal:** Ensure `generate_report_updated.py` can process both old and new format data.

**Implementation:**
1. **Add Column Detection**:
   ```python
   def detect_evaluation_version(df: pd.DataFrame) -> str:
       """Detect if evaluation data is v1 (score) or v2 (A/B verdict)."""
       if 'verdict_token' in df.columns and 'confidence_fake' in df.columns:
           return 'v2'
       elif 'score_example' in df.columns:
           return 'v1'
       else:
           raise ValueError("Unknown evaluation data format")
   ```

2. **Normalize Data Before Processing**:
   ```python
   # In generate_report_updated.py, after loading predictions_df:
   version = detect_evaluation_version(predictions_df)

   if version == 'v1':
       # Add computed columns for v2 compatibility
       predictions_df['confidence_fake'] = predictions_df['score_example'] / 10.0
       predictions_df['verdict_token'] = predictions_df['consensus_label'].map({
           'AI Generated': 'B',
           'Real': 'A'
       })
       predictions_df['tier'] = predictions_df['score_example'].apply(
           lambda s: 'Deepfake' if s >= 9 else ('Suspicious' if s >= 5 else 'Authentic')
       )
   ```

3. **Add Tier-Based Analysis**:
   - New visualization: "Three-Tier Distribution" (Authentic vs Suspicious vs Deepfake)
   - Confidence distribution histogram
   - Comparison of tier predictions across models

### Phase 4: Migration Utility

**Goal:** Convert old evaluation results to new format for unified analysis.

**Create:** `migrate_evaluation_results.py`

```python
"""
Migration utility to convert old score-based evaluation results
to new A/B verdict format.
"""

import pandas as pd
import openpyxl
from pathlib import Path

def migrate_evaluation_file(input_path: str, output_path: str):
    """
    Migrate old evaluation Excel file to new format.

    Args:
        input_path: Path to old evaluation_YYYYMMDD_HHMMSS.xlsx
        output_path: Path to save migrated file
    """
    # Load both sheets
    metrics_df = pd.read_excel(input_path, sheet_name='metrics')
    predictions_df = pd.read_excel(input_path, sheet_name='predictions')

    # Add new columns to predictions
    predictions_df['verdict_token'] = predictions_df['consensus_label'].map({
        'AI Generated': 'B',
        'Real': 'A'
    })

    predictions_df['confidence_fake'] = predictions_df['score_example'] / 10.0

    def score_to_tier(score):
        if score >= 9:
            return 'Deepfake'
        elif score >= 5:
            return 'Suspicious'
        else:
            return 'Authentic'

    predictions_df['tier'] = predictions_df['score_example'].apply(score_to_tier)

    # Metrics sheet remains unchanged

    # Write to new file
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='metrics', index=False)
        predictions_df.to_excel(writer, sheet_name='predictions', index=False)

    print(f"✓ Migrated {input_path} → {output_path}")
    print(f"  Added columns: verdict_token, confidence_fake, tier")

def migrate_all_results(results_dir: str = 'results'):
    """Migrate all evaluation files in results directory."""
    results_path = Path(results_dir)
    for xlsx_file in results_path.glob('evaluation_*.xlsx'):
        if '_migrated' in xlsx_file.name:
            continue  # Skip already migrated files

        output_name = xlsx_file.stem + '_migrated.xlsx'
        output_path = results_path / output_name

        migrate_evaluation_file(str(xlsx_file), str(output_path))

if __name__ == '__main__':
    migrate_all_results()
```

## Implementation Checklist

### Step 1: Core Functionality
- [ ] Create `normalize_verdict()` in `shared_functions.py`
- [ ] Create `analyze_single_image_v2()` using `OSINTDetector`
- [ ] Add format detection logic
- [ ] Test with sample images (both old and new paths)

### Step 2: Batch Evaluation Integration
- [ ] Update `app.py` Tab 2 to add "Verdict System" toggle
- [ ] Integrate `OSINTDetector` initialization in batch mode
- [ ] Modify Excel export to include new columns
- [ ] Add backward compatibility for score_example computation
- [ ] Test batch processing with multiple images

### Step 3: Report Generator Compatibility
- [ ] Add version detection to `generate_report_updated.py`
- [ ] Implement data normalization for v1 files
- [ ] Add tier-based visualizations
- [ ] Test with old evaluation files (evaluation_20251126_204434.xlsx, evaluation_20251202_213404.xlsx)

### Step 4: Migration and Testing
- [ ] Create `migrate_evaluation_results.py`
- [ ] Run migration on existing results files
- [ ] Compare old vs migrated file metrics
- [ ] Generate reports from both v1 and v2 data
- [ ] Verify metrics consistency (TP/TN/FP/FN should match)

### Step 5: Documentation
- [ ] Update README with new verdict system explanation
- [ ] Document migration process
- [ ] Add examples of old vs new data format
- [ ] Create troubleshooting guide

## Testing Strategy

### Unit Tests
1. **Verdict Normalization**:
   - Score 10 → confidence_fake=1.0, tier="Deepfake", classification="AI Generated"
   - Score 0 → confidence_fake=0.0, tier="Authentic", classification="Real"
   - Score 5 → confidence_fake=0.5, tier="Suspicious", classification="AI Generated"

2. **Token Parsing**:
   - Token "A" → classification="Real"
   - Token "B" → classification="AI Generated"
   - Missing token → confidence=0.5 (uncertain)

### Integration Tests
1. **Batch Evaluation**:
   - Run 5 images with old system → export Excel
   - Run same 5 images with new system → export Excel
   - Compare metrics (should have similar accuracy if thresholds aligned)

2. **Report Generation**:
   - Load old evaluation file → generate report
   - Load migrated file → generate report
   - Compare visualizations (charts should match)

### Compatibility Tests
1. **Mixed Data**:
   - Merge v1 and v2 evaluation files
   - Generate combined report
   - Verify no errors and correct data handling

## Rollout Plan

### Phase 1: Development (Week 1)
- Implement core normalization functions
- Add detector integration to batch evaluation
- Test with small datasets

### Phase 2: Testing (Week 2)
- Run parallel evaluations (old vs new system)
- Validate metric consistency
- Fix any discrepancies

### Phase 3: Migration (Week 3)
- Migrate existing results files
- Generate comparison reports
- Validate against original reports

### Phase 4: Deployment (Week 4)
- Update Docker container with new system
- Set new system as default (with old system still available)
- Update documentation
- Train users on new confidence/tier interpretation

## Benefits of New System

1. **Probabilistic Confidence**: More nuanced than binary score threshold
2. **Three-Tier Classification**: Better risk stratification (Authentic / Suspicious / Deepfake)
3. **Logprob-Based**: More reliable than text parsing
4. **Forensic Integration**: Optional ELA/FFT artifacts for enhanced detection
5. **Context-Aware**: OSINT-specific protocols for military/disaster/propaganda imagery
6. **Prompt Engineering**: YAML-based prompts for easy iteration

## Backward Compatibility Guarantees

1. **Existing Data**: All old evaluation files remain readable
2. **Metrics Calculation**: TP/TN/FP/FN logic unchanged
3. **Report Generation**: Works with both v1 and v2 data
4. **Excel Schema**: New columns added, old columns preserved
5. **API**: Old `analyze_single_image()` function preserved

## Migration Path for Users

### Option 1: Keep Old System
- No action required
- Continue using score-based evaluation
- All existing workflows remain functional

### Option 2: Migrate to New System
1. Run `migrate_evaluation_results.py` on results folder
2. Regenerate reports with migrated data
3. Switch to "A/B Verdict System" in app.py
4. Future evaluations use new system

### Option 3: Hybrid Approach
- Use old system for datasets requiring score interpretation
- Use new system for new evaluations requiring confidence levels
- Report generator handles both transparently

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Metric discrepancy between systems | High | Thorough testing, threshold calibration |
| Data corruption during migration | High | Backup files before migration, validation checks |
| Performance degradation (detector is slower) | Medium | Batch processing optimization, progress tracking |
| User confusion about new format | Medium | Clear documentation, training materials |
| Incompatibility with external tools | Low | Maintain score_example column for legacy tools |

## Success Criteria

1. ✅ All existing evaluation files can be processed without errors
2. ✅ New evaluations produce valid Excel files with all columns
3. ✅ Report generator works with both v1 and v2 data
4. ✅ Metrics (accuracy, precision, recall, F1) consistent between systems
5. ✅ Migration completes without data loss
6. ✅ Documentation covers all use cases
7. ✅ Docker container builds and runs successfully

## Timeline

- **Week 1**: Implementation of core functions and integration
- **Week 2**: Testing and validation
- **Week 3**: Migration of existing data and report generation
- **Week 4**: Documentation, deployment, and user training

**Total Estimated Time:** 4 weeks
