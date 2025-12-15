# SPAI Integration Implementation Plan

## Overview

Replace the current forensic analysis system (ELA/FFT) with SPAI (Spectral AI-Generated Image Detector) to provide more reliable deepfake detection with two operational modes:

1. **Standalone SPAI Mode**: SPAI acts as the sole detector (fast, reliable, no VLM needed)
2. **SPAI-Assisted VLM Mode**: SPAI provides spectral analysis to enhance VLM reasoning

## Current System Architecture

### Forensics Module (forensics.py - 645 lines)
```python
class ArtifactGenerator:
    - generate_ela() → ELA map (PNG bytes)
    - generate_fft_preprocessed() → FFT spectrum (PNG bytes) + metrics dict
    - compute_ela_variance() → float (compression uniformity metric)
    - _build_forensic_report() → text report with ELA/FFT interpretation
```

### Detector Integration (detector.py)
```python
class OSINTDetector:
    __init__():
        self.artifact_gen = ArtifactGenerator()

    detect():
        # Stage 0: Metadata extraction
        metadata_report, auto_fail = _check_metadata()

        # Stage 1: Generate forensic artifacts
        ela_bytes = artifact_gen.generate_ela(image_bytes)
        fft_bytes, fft_metrics = artifact_gen.generate_fft_preprocessed(image_bytes)
        ela_variance = artifact_gen.compute_ela_variance(ela_bytes)
        forensic_report = _build_forensic_report(metadata, ela_variance, fft_metrics)

        # Stage 2: VLM analysis with forensics
        if send_forensics:
            VLM receives: [forensic_report_text, original_image, ELA_image, FFT_image]
        else:
            VLM receives: [original_image]

        # Stage 3: Verdict extraction (A/B with logprobs)

        return {
            "tier": str,
            "confidence": float,
            "reasoning": str,
            "forensic_report": str,  # ← Replace with SPAI analysis
            "verdict_token": str
        }
```

## SPAI Integration Architecture

### New Module: spai_detector.py

```python
class SPAIDetector:
    """
    SPAI-based deepfake detection with dual operational modes.

    Modes:
        - standalone: SPAI provides final verdict (no VLM needed)
        - assisted: SPAI provides spectral analysis for VLM reasoning
    """

    def __init__(
        self,
        config_path: str = "spai/configs/spai.yaml",
        weights_path: str = "spai/weights/spai.pth",
        device: str = "cuda"
    ):
        """Initialize SPAI model."""
        self.config = get_config({"cfg": config_path, "batch_size": 1, "opts": []})
        self.model = build_cls_model(self.config)
        self.model.to(device)
        load_pretrained(self.config, self.model, logger, checkpoint_path=weights_path)
        self.model.eval()
        self.transform = build_transform(self.config, is_train=False)

    def analyze(
        self,
        image_bytes: bytes,
        generate_heatmap: bool = True
    ) -> Dict:
        """
        Run SPAI analysis on image.

        Args:
            image_bytes: Input image as bytes
            generate_heatmap: If True, generate attention overlay

        Returns:
            {
                "spai_score": float,        # 0.0-1.0 (AI-generated probability)
                "spai_prediction": str,     # "Real" or "AI Generated"
                "spai_confidence": float,   # 0.0-1.0 (confidence in prediction)
                "heatmap_bytes": bytes,     # PNG overlay showing attention (if requested)
                "analysis_text": str        # Human-readable interpretation
            }
        """
        # Load and preprocess image
        pil_image = Image.open(io.BytesIO(image_bytes))
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(tensor)
            score = torch.sigmoid(output).item()  # 0-1 probability

        # Generate prediction
        prediction = "AI Generated" if score >= 0.5 else "Real"
        confidence = score if score >= 0.5 else (1.0 - score)

        # Generate heatmap if requested
        heatmap_bytes = None
        if generate_heatmap:
            heatmap_bytes = self._generate_attention_heatmap(tensor, pil_image)

        # Create human-readable analysis
        analysis_text = self._format_analysis(score, prediction, confidence)

        return {
            "spai_score": score,
            "spai_prediction": prediction,
            "spai_confidence": confidence,
            "heatmap_bytes": heatmap_bytes,
            "analysis_text": analysis_text
        }

    def _generate_attention_heatmap(
        self,
        tensor: torch.Tensor,
        original_image: Image.Image
    ) -> bytes:
        """
        Generate attention overlay heatmap.

        Uses model's internal attention weights to highlight suspicious regions.
        Blends heatmap with original image (60% original, 40% heatmap).

        Returns:
            PNG-encoded bytes of the blended overlay
        """
        # Extract attention weights from model
        attention_maps = self.model.get_attention_maps(tensor)

        # Aggregate attention across layers
        aggregated_attention = attention_maps.mean(dim=0)  # Average over layers

        # Resize to original image size
        heatmap = F.interpolate(
            aggregated_attention.unsqueeze(0).unsqueeze(0),
            size=original_image.size[::-1],
            mode='bilinear'
        ).squeeze()

        # Normalize to 0-255 and apply colormap
        heatmap_np = (heatmap.cpu().numpy() * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

        # Blend with original
        original_np = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        blended = cv2.addWeighted(original_np, 0.6, heatmap_colored, 0.4, 0)

        # Encode as PNG
        success, png_bytes = cv2.imencode('.png', blended)
        return png_bytes.tobytes()

    def _format_analysis(
        self,
        score: float,
        prediction: str,
        confidence: float
    ) -> str:
        """Format SPAI analysis as human-readable text."""
        tier = self._get_tier(score)

        analysis = f"""--- SPAI SPECTRAL ANALYSIS ---

Prediction: {prediction}
SPAI Score: {score:.3f} (0=Real, 1=AI-Generated)
Confidence: {confidence:.1%}
Risk Tier: {tier}

Spectral Analysis Summary:
The image's frequency spectrum was analyzed using masked feature modeling
with Vision Transformer architecture. The spectral reconstruction similarity
score indicates {self._get_likelihood_text(score)}.

"""

        if score >= 0.9:
            analysis += "🚨 HIGH CONFIDENCE AI-GENERATED: Strong spectral artifacts detected.\n"
        elif score >= 0.7:
            analysis += "⚠️ LIKELY AI-GENERATED: Moderate spectral inconsistencies found.\n"
        elif score >= 0.5:
            analysis += "⚠️ SUSPICIOUS: Subtle spectral anomalies present.\n"
        elif score >= 0.3:
            analysis += "✓ LIKELY REAL: Spectral patterns consistent with real images.\n"
        else:
            analysis += "✅ HIGH CONFIDENCE REAL: Natural spectral distribution detected.\n"

        analysis += "\nNote: SPAI analyzes frequency-domain patterns invisible to human perception."

        return analysis

    def _get_tier(self, score: float) -> str:
        """Map SPAI score to three-tier system."""
        if score >= 0.9:
            return "Deepfake (High Confidence)"
        elif score >= 0.5:
            return "Suspicious"
        else:
            return "Authentic"

    def _get_likelihood_text(self, score: float) -> str:
        """Convert score to likelihood text."""
        if score >= 0.9:
            return "very high likelihood of AI generation"
        elif score >= 0.7:
            return "high likelihood of AI generation"
        elif score >= 0.5:
            return "moderate likelihood of AI generation"
        elif score >= 0.3:
            return "low likelihood of AI generation"
        else:
            return "very low likelihood of AI generation"
```

### Updated: detector.py

```python
class OSINTDetector:
    """
    Two-stage OSINT-specialized deepfake detector with SPAI integration.

    Detection Modes:
        - "spai_standalone": SPAI provides final verdict (no VLM analysis)
        - "spai_assisted": SPAI provides spectral analysis, VLM provides reasoning
        - "vlm_only": Legacy mode without SPAI (for comparison)
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "dummy",
        context: str = "auto",
        watermark_mode: str = "ignore",
        provider: str = "vllm",
        prompts_path: str = "prompts.yaml",
        detection_mode: str = "spai_assisted"  # NEW PARAMETER
    ):
        """
        Initialize OSINT detector with SPAI integration.

        Args:
            detection_mode: One of:
                - "spai_standalone": SPAI only (fast, no VLM)
                - "spai_assisted": SPAI + VLM (comprehensive)
                - "vlm_only": VLM only (legacy, no SPAI)
        """
        self.model_name = model_name
        self.context = context
        self.watermark_mode = watermark_mode
        self.provider = provider
        self.detection_mode = detection_mode

        # Load prompts from YAML
        self.prompts = self._load_prompts(prompts_path)

        # Initialize SPAI detector if needed
        if detection_mode in ["spai_standalone", "spai_assisted"]:
            from spai_detector import SPAIDetector
            self.spai = SPAIDetector()
        else:
            self.spai = None

        # Initialize VLM client if needed
        if detection_mode in ["spai_assisted", "vlm_only"]:
            if provider in ["vllm", "openai"]:
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    timeout=180.0,
                    max_retries=0
                )
            else:
                self.client = get_cloud_adapter(provider, model_name, api_key, base_url)
        else:
            self.client = None

        # Legacy forensics for vlm_only mode
        if detection_mode == "vlm_only":
            from forensics import ArtifactGenerator
            self.artifact_gen = ArtifactGenerator()
        else:
            self.artifact_gen = None

    def detect(
        self,
        image_bytes: bytes,
        debug: bool = False,
        send_forensics: bool = True
    ) -> Dict:
        """
        Perform deepfake detection using configured mode.

        Returns:
            {
                "tier": str,
                "confidence": float,
                "reasoning": str,
                "forensic_report": str,  # Now contains SPAI analysis or legacy forensics
                "metadata_auto_fail": bool,
                "raw_logits": dict,
                "verdict_token": str,
                "spai_score": float (if SPAI used),
                "heatmap_bytes": bytes (if SPAI used)
            }
        """
        # Stage 0: Metadata extraction (always performed)
        metadata_dict, metadata_report, auto_fail = self._check_metadata(image_bytes)

        if auto_fail:
            return {
                "tier": "Deepfake",
                "confidence": 1.0,
                "reasoning": "AI generation tool detected in image metadata",
                "forensic_report": metadata_report,
                "metadata_auto_fail": True,
                "raw_logits": {"real": -100.0, "fake": 0.0},
                "verdict_token": "B"
            }

        # Route to appropriate detection pipeline
        if self.detection_mode == "spai_standalone":
            return self._detect_spai_standalone(image_bytes, metadata_report)
        elif self.detection_mode == "spai_assisted":
            return self._detect_spai_assisted(image_bytes, metadata_report, send_forensics)
        else:  # vlm_only
            return self._detect_vlm_only(image_bytes, metadata_report, send_forensics)

    def _detect_spai_standalone(
        self,
        image_bytes: bytes,
        metadata_report: str
    ) -> Dict:
        """
        SPAI-only detection (no VLM analysis).

        Fast, reliable, suitable for batch processing.
        """
        # Run SPAI analysis
        spai_result = self.spai.analyze(image_bytes, generate_heatmap=True)

        # Map SPAI prediction to tier
        score = spai_result["spai_score"]
        if score >= 0.9:
            tier = "Deepfake"
        elif score >= 0.5:
            tier = "Suspicious"
        else:
            tier = "Authentic"

        # Determine verdict token (A=Real, B=Fake)
        verdict_token = "B" if score >= 0.5 else "A"

        # Confidence = distance from threshold
        confidence_fake = score

        # Combine metadata and SPAI analysis
        forensic_report = f"{metadata_report}\n\n{spai_result['analysis_text']}"

        return {
            "tier": tier,
            "confidence": confidence_fake,
            "reasoning": f"SPAI Standalone Detection\n\n{spai_result['analysis_text']}",
            "forensic_report": forensic_report,
            "metadata_auto_fail": False,
            "raw_logits": {
                "real": np.log(1 - score) if score < 1.0 else -100.0,
                "fake": np.log(score) if score > 0.0 else -100.0
            },
            "verdict_token": verdict_token,
            "spai_score": score,
            "heatmap_bytes": spai_result["heatmap_bytes"]
        }

    def _detect_spai_assisted(
        self,
        image_bytes: bytes,
        metadata_report: str,
        send_forensics: bool
    ) -> Dict:
        """
        SPAI + VLM detection (comprehensive analysis).

        SPAI provides spectral analysis, VLM provides reasoning and final verdict.
        """
        # Run SPAI analysis
        spai_result = self.spai.analyze(image_bytes, generate_heatmap=True)

        # Build forensic report with SPAI analysis
        forensic_report = f"{metadata_report}\n\n{spai_result['analysis_text']}"

        # Get system prompt
        system_prompt = self._get_system_prompt(include_forensic_context=send_forensics)

        # Run VLM analysis
        analysis, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens = \
            self._two_stage_classification_with_spai(
                image_bytes,
                spai_result["heatmap_bytes"],
                forensic_report,
                spai_result["spai_score"],
                system_prompt,
                send_forensics
            )

        # Determine tier
        tier = self._classify_tier(verdict_result["confidence"])

        return {
            "tier": tier,
            "confidence": verdict_result["confidence"],
            "reasoning": analysis,
            "forensic_report": forensic_report,
            "metadata_auto_fail": False,
            "raw_logits": verdict_result["raw_logits"],
            "verdict_token": verdict_result["token"],
            "spai_score": spai_result["spai_score"],
            "heatmap_bytes": spai_result["heatmap_bytes"],
            "req1_time": req1_time,
            "req2_time": req2_time,
            "req1_tokens": req1_tokens,
            "req2_tokens": req2_tokens
        }

    def _detect_vlm_only(
        self,
        image_bytes: bytes,
        metadata_report: str,
        send_forensics: bool
    ) -> Dict:
        """
        Legacy VLM-only detection (no SPAI).

        Uses original ELA/FFT forensics pipeline.
        """
        # Generate legacy forensic artifacts
        ela_bytes = self.artifact_gen.generate_ela(image_bytes)
        fft_bytes, fft_metrics = self.artifact_gen.generate_fft_preprocessed(image_bytes)
        ela_variance = self.artifact_gen.compute_ela_variance(ela_bytes)

        forensic_report = self._build_forensic_report(
            metadata_report,
            ela_variance,
            fft_metrics
        )

        system_prompt = self._get_system_prompt(include_forensic_context=send_forensics)

        analysis, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens = \
            self._two_stage_classification(
                image_bytes,
                ela_bytes,
                fft_bytes,
                forensic_report,
                system_prompt,
                send_forensics
            )

        tier = self._classify_tier(verdict_result["confidence"])

        return {
            "tier": tier,
            "confidence": verdict_result["confidence"],
            "reasoning": analysis,
            "forensic_report": forensic_report,
            "metadata_auto_fail": False,
            "raw_logits": verdict_result["raw_logits"],
            "verdict_token": verdict_result["token"]
        }

    def _two_stage_classification_with_spai(
        self,
        original_bytes: bytes,
        heatmap_bytes: bytes,
        forensic_report: str,
        spai_score: float,
        system_prompt: str,
        send_forensics: bool = True
    ) -> Tuple[str, Dict, float, float, int, int]:
        """
        Two-stage VLM classification with SPAI assistance.

        Stage 2: VLM analyzes original image + SPAI heatmap + SPAI analysis
        Stage 3: VLM provides binary verdict (A/B)
        """
        original_uri = f"data:image/png;base64,{base64.b64encode(original_bytes).decode()}"

        user_content = []

        if send_forensics:
            # Include SPAI analysis and heatmap
            heatmap_uri = f"data:image/png;base64,{base64.b64encode(heatmap_bytes).decode()}"

            user_content.extend([
                {"type": "text", "text": "--- SPAI SPECTRAL ANALYSIS ---"},
                {"type": "text", "text": forensic_report},
                {"type": "text", "text": f"\nSPAI Score: {spai_score:.3f} (0=Real, 1=AI-Generated)\n"},
                {"type": "text", "text": "--- ORIGINAL IMAGE ---"},
                {"type": "image_url", "image_url": {"url": original_uri}},
                {"type": "text", "text": "--- SPAI ATTENTION HEATMAP ---"},
                {"type": "image_url", "image_url": {"url": heatmap_uri}},
                {
                    "type": "text",
                    "text": (
                        "--- ANALYSIS INSTRUCTIONS ---\n"
                        "This is the image to be analysed. You have been provided with:\n\n"
                        "1. **SPAI Spectral Analysis**: Frequency-domain analysis showing AI generation likelihood\n"
                        f"   - SPAI Score: {spai_score:.3f} (scores > 0.5 indicate AI generation)\n"
                        "   - This is a CVPR2025 state-of-the-art spectral detector with ~95% accuracy\n\n"
                        "2. **SPAI Attention Heatmap**: Red regions indicate areas with spectral anomalies\n"
                        "   - Overlaid on original image for visual reference\n"
                        "   - Highlights suspicious frequency patterns\n\n"
                        f"{self.prompts['analysis_instructions']['visual_analysis']}\n\n"
                        "3. **Cross-Reference with SPAI**:\n"
                        "   - Do your visual findings align with SPAI's spectral analysis?\n"
                        "   - Are the attention hotspots correlated with visual artifacts you detected?\n"
                        f"   - SPAI indicates: {'AI-generated' if spai_score >= 0.5 else 'Real'}\n\n"
                        f"4. **Watermark Analysis**: {self._get_watermark_instruction()}\n\n"
                        "Provide comprehensive reasoning combining both visual analysis and spectral findings."
                    )
                }
            ])
        else:
            # Simplified: just original image (no SPAI assistance)
            user_content.extend([
                {"type": "text", "text": "--- IMAGE TO ANALYZE ---"},
                {"type": "image_url", "image_url": {"url": original_uri}},
                {
                    "type": "text",
                    "text": (
                        "--- ANALYSIS INSTRUCTIONS ---\n"
                        f"{self.prompts['analysis_instructions']['visual_analysis']}\n\n"
                        f"**Watermark Analysis**: {self._get_watermark_instruction()}\n\n"
                        "Provide your reasoning for whether this image is authentic or AI-generated."
                    )
                }
            ])

        # Request 1: Analysis (Stage 2)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        req1_start = time.time()
        try:
            response_1 = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=2000
            )
            req1_time = time.time() - req1_start
            analysis_text = response_1.choices[0].message.content
            req1_tokens = len(forensic_report) // 4 + 765 * 2 + 50  # 2 images
        except Exception as e:
            req1_time = time.time() - req1_start
            analysis_text = f"VLM analysis failed: {type(e).__name__}: {str(e)}"
            req1_tokens = 0

        # Request 2: Verdict (Stage 3)
        verdict_prompt = f"""Based on your analysis above, provide your final verdict.

(A) Real (Authentic Capture)
(B) Fake (AI Generated/Manipulated)

Answer with ONLY the single letter A or B."""

        messages.append({"role": "user", "content": verdict_prompt})

        req2_start = time.time()
        try:
            response_2 = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=5
            )
            req2_time = time.time() - req2_start
            verdict_result = self._parse_verdict(response_2)
            req2_tokens = len(verdict_prompt.split())
        except Exception as e:
            req2_time = time.time() - req2_start
            verdict_result = {
                "token": None,
                "confidence": 0.5,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "error": f"Stage 3 (Verdict) failed: {type(e).__name__}: {str(e)}"
            }
            req2_tokens = 0

        return analysis_text, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens
```

### Updated: prompts.yaml

Add SPAI-specific prompts:

```yaml
spai_prompts:
  standalone_reasoning: |
    This verdict is based on SPAI (Spectral AI-Generated Image Detector), a CVPR2025
    state-of-the-art method that analyzes frequency-domain patterns using masked feature
    modeling with Vision Transformers.

    SPAI achieves ~95% accuracy on benchmark datasets by detecting spectral artifacts
    invisible to human perception. The model was trained on diverse AI generators
    (GANs, diffusion models, etc.) and generalizes well to unseen generators.

  assisted_context: |
    You are receiving assistance from SPAI, a spectral AI-generated image detector with
    ~95% accuracy on CVPR benchmarks. SPAI analyzes frequency-domain patterns that are
    invisible to human perception but characteristic of AI generation.

    Your task is to combine SPAI's spectral analysis with your visual reasoning to
    provide a comprehensive verdict. Consider:
    - Do your visual findings support SPAI's spectral assessment?
    - Are attention hotspots correlated with visual artifacts?
    - Does the spectral score align with semantic coherence?
```

### Updated: app.py

Add mode selection UI:

```python
# In app.py batch evaluation section
st.subheader("Detection Mode")
detection_mode = st.radio(
    "Select detection mode",
    options=[
        "spai_standalone",
        "spai_assisted",
        "vlm_only"
    ],
    format_func=lambda x: {
        "spai_standalone": "🚀 SPAI Standalone (Fast, no VLM)",
        "spai_assisted": "🔬 SPAI-Assisted VLM (Comprehensive)",
        "vlm_only": "🧠 VLM Only (Legacy, no SPAI)"
    }[x],
    index=1,  # Default to spai_assisted
    help="""
    - SPAI Standalone: Fast spectral analysis only (~50ms/image)
    - SPAI-Assisted VLM: Spectral + semantic analysis (~3s/image)
    - VLM Only: Legacy ELA/FFT forensics + VLM (~3s/image)
    """
)

# Pass detection_mode to analyzer
detector = OSINTDetector(
    base_url=model_config['base_url'],
    model_name=model_config['model_name'],
    api_key=model_config['api_key'],
    context=eval_context,
    watermark_mode=watermark_mode,
    provider=model_config['provider'],
    detection_mode=detection_mode  # NEW
)
```

## Implementation Phases

### Phase 1: SPAI Detector Module ✅
**Tasks:**
1. Create `spai_detector.py` with `SPAIDetector` class
2. Implement `analyze()` method for inference
3. Implement `_generate_attention_heatmap()` for overlay
4. Implement `_format_analysis()` for human-readable output
5. Test standalone SPAI inference on sample images

**Deliverables:**
- `spai_detector.py` (new file, ~300 lines)
- Unit tests for SPAI inference

### Phase 2: Detector Integration ✅
**Tasks:**
1. Add `detection_mode` parameter to `OSINTDetector.__init__()`
2. Implement `_detect_spai_standalone()` method
3. Implement `_detect_spai_assisted()` method
4. Update `_detect_vlm_only()` to preserve legacy behavior
5. Implement `_two_stage_classification_with_spai()` method
6. Update return dictionaries to include `spai_score` and `heatmap_bytes`

**Deliverables:**
- Updated `detector.py` (~700 lines)
- Backward compatibility with vlm_only mode

### Phase 3: Prompt Engineering ✅
**Tasks:**
1. Add `spai_prompts` section to `prompts.yaml`
2. Update visual analysis instructions for SPAI context
3. Create SPAI-specific interpretation guidance
4. Test prompt effectiveness with sample images

**Deliverables:**
- Updated `prompts.yaml`
- Prompt testing report

### Phase 4: UI Integration ✅
**Tasks:**
1. Update `app.py` single image analysis to support mode selection
2. Update batch evaluation UI with detection_mode radio buttons
3. Add SPAI heatmap display in results
4. Update Excel export to include SPAI columns
5. Update confusion matrix display for SPAI standalone mode

**Deliverables:**
- Updated `app.py`
- UI screenshots showing mode selection

### Phase 5: Batch Evaluation ✅
**Tasks:**
1. Update `shared_functions.py` to support detection_mode
2. Modify `analyze_single_image()` to use new detector API
3. Update Excel schema: add `spai_score`, `spai_prediction`, `detection_mode`
4. Update report generator to handle SPAI columns
5. Create comparison scripts (SPAI vs VLM vs SPAI+VLM)

**Deliverables:**
- Updated `shared_functions.py`
- Updated `generate_report_updated.py`
- Comparison evaluation reports

### Phase 6: Docker & Dependencies ✅
**Tasks:**
1. Add SPAI requirements to `requirements.txt`:
   - torch>=2.0.0
   - torchvision>=0.15.0
   - timm>=0.9.0
   - (others from spai/requirements.txt)
2. Update `Dockerfile` to:
   - Copy `spai/` directory
   - Copy `spai_detector.py`
   - Download SPAI weights (340MB)
   - Install SPAI dependencies
3. Update `docker-compose.yml` to mount SPAI weights
4. Test Docker build and runtime

**Deliverables:**
- Updated `requirements.txt`
- Updated `Dockerfile`
- Updated `docker-compose.yml`
- Docker testing report

### Phase 7: Testing & Validation ✅
**Tasks:**
1. Unit tests for `spai_detector.py`
2. Integration tests for all three modes
3. Performance benchmarking (speed, accuracy)
4. Comparison evaluation on existing test set:
   - Run all 3 modes on same images
   - Compare metrics (TP/TN/FP/FN)
   - Generate comparison report
5. Edge case testing (corrupted images, extreme resolutions)

**Deliverables:**
- Test suite
- Performance benchmarks
- Comparison evaluation report

### Phase 8: Documentation ✅
**Tasks:**
1. Create `SPAI_INTEGRATION.md` guide
2. Update `README.md` with SPAI mode descriptions
3. Create mode selection flowchart
4. Document SPAI score interpretation
5. Create troubleshooting guide

**Deliverables:**
- Complete documentation set
- User guide for mode selection

## Expected Performance

### Speed Comparison

| Mode | Processing Time | Use Case |
|------|----------------|----------|
| **SPAI Standalone** | ~50ms | Batch screening, real-time |
| **SPAI-Assisted VLM** | ~3s | Comprehensive analysis |
| **VLM Only** | ~3s | Legacy comparison |

### Accuracy Expectations

| Mode | Expected Accuracy | Strengths |
|------|------------------|-----------|
| **SPAI Standalone** | ~95% | Fast, reliable, frequency-based |
| **SPAI-Assisted VLM** | ~97%+ | Best accuracy, multi-modal |
| **VLM Only** | ~90% | Semantic reasoning, explainable |

### Resource Requirements

| Component | GPU Memory | Disk Space |
|-----------|-----------|------------|
| SPAI Model | ~2GB | 340MB |
| VLM (vLLM) | ~8GB | Variable |
| **Total** | ~10GB | ~1GB |

## Migration from Current System

### Backward Compatibility

✅ **Preserved:**
- All existing evaluation files remain valid
- `vlm_only` mode provides identical behavior to current system
- Metrics calculation (TP/TN/FP/FN) unchanged
- Excel export schema backward compatible

✅ **Enhanced:**
- New `detection_mode` column in Excel exports
- Optional `spai_score` and `spai_prediction` columns
- Heatmap bytes stored in results

### Migration Steps

1. **Install SPAI dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download SPAI weights:**
   ```bash
   cd spai/weights
   wget https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view?usp=sharing
   ```

3. **Test SPAI standalone:**
   ```bash
   streamlit run app.py
   # Select "SPAI Standalone" mode
   ```

4. **Compare modes on test set:**
   ```python
   # Run batch evaluation with all 3 modes
   # Compare results
   ```

5. **Choose default mode:**
   - Production: `spai_assisted` (best accuracy)
   - Batch screening: `spai_standalone` (fastest)
   - Legacy comparison: `vlm_only`

## Risk Mitigation

### Potential Issues

1. **SPAI weights not found**
   - Solution: Graceful fallback to vlm_only mode
   - Warning message to user

2. **GPU out of memory**
   - Solution: Detect GPU memory, use CPU fallback for SPAI
   - Warning about slower inference

3. **Different verdicts between SPAI and VLM**
   - Expected behavior in spai_assisted mode
   - VLM has final say, uses SPAI as guidance

4. **Attention heatmap generation fails**
   - Solution: Continue without heatmap
   - Log warning, return None for heatmap_bytes

5. **SPAI library version conflicts**
   - Solution: Isolate SPAI imports in spai_detector.py
   - Use try/except for import failures

### Rollback Plan

If SPAI integration causes issues:

1. **Immediate rollback:**
   ```bash
   git checkout feature/osint-detection
   docker compose up --build -d
   ```

2. **Partial rollback:**
   - Set default detection_mode to "vlm_only"
   - SPAI remains available but not default

3. **Gradual migration:**
   - Phase 1: Deploy with vlm_only as default
   - Phase 2: Enable spai_assisted for power users
   - Phase 3: Make spai_assisted default after validation

## Success Criteria

✅ **Phase 1 Complete:**
- SPAI standalone inference works
- Accuracy >= 95% on test set
- Speed <= 100ms per image

✅ **Phase 2 Complete:**
- All 3 modes functional in detector.py
- Backward compatibility preserved
- Unit tests pass

✅ **Phase 3 Complete:**
- VLM correctly interprets SPAI analysis
- Prompts generate coherent reasoning
- Accuracy improvement in spai_assisted mode

✅ **Phase 4-8 Complete:**
- Full UI integration
- Batch evaluation working
- Docker deployment successful
- Documentation complete

## Timeline

- **Phase 1-2**: 2 days (Core integration)
- **Phase 3**: 1 day (Prompt engineering)
- **Phase 4-5**: 2 days (UI and batch eval)
- **Phase 6**: 1 day (Docker)
- **Phase 7**: 2 days (Testing)
- **Phase 8**: 1 day (Documentation)

**Total**: ~9 days for complete integration

## Conclusion

This integration plan provides:
1. **Dual-mode detection** (standalone + assisted)
2. **Backward compatibility** (vlm_only mode)
3. **Performance improvement** (95%+ accuracy)
4. **Speed optimization** (50ms standalone)
5. **Enhanced explainability** (attention heatmaps)

The modular design allows gradual adoption and easy rollback if needed.
