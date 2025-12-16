# SPAI Integration Implementation Plan (Simplified)

## Overview

**Completely replace** the current forensic analysis system (ELA/FFT) with SPAI (Spectral AI-Generated Image Detector):

1. **Standalone SPAI Mode**: SPAI acts as the sole detector (fast, no VLM needed)
2. **SPAI-Assisted VLM Mode**: SPAI provides spectral analysis + attention overlay for VLM reasoning

**Critical Changes:**
- ✅ SPAI replaces all ELA/FFT forensics
- ✅ `forensics.py` module **deleted** (645 lines)
- ✅ No `vlm_only` mode - this is a clean break from legacy forensics
- ✅ Transparent attention overlay blending (60% original, 40% heatmap) passed to VLM

## Architecture

### Two Detection Modes Only

#### Mode 1: SPAI Standalone
```
Input: image_bytes
→ SPAI inference (~50ms)
→ Output: {score, prediction, tier, heatmap, analysis_text}
```

#### Mode 2: SPAI-Assisted VLM
```
Input: image_bytes
→ SPAI inference (~50ms)
→ Generate blended attention overlay (alpha=0.6)
→ VLM receives: [original_image, blended_overlay, SPAI_analysis, SPAI_score]
→ VLM analysis (~3s)
→ VLM verdict with logprobs
→ Output: {score, prediction, tier, reasoning, heatmap}
```

### Key Simplifications

1. **No ELA/FFT Generation**
   - Delete `forensics.py` entirely
   - Remove all `ArtifactGenerator` references
   - No ELA variance calculations
   - No FFT peak detection

2. **Single Overlay Type**
   - Only SPAI attention heatmap
   - Blended with `cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)`
   - This matches `spai/app.py` implementation

3. **Two-Mode System**
   - `detection_mode="spai_standalone"` or `"spai_assisted"`
   - No legacy mode = simpler code
   - No backward compatibility needed

## Implementation

### Phase 1: Create spai_detector.py

```python
import torch
import cv2
import numpy as np
from PIL import Image
import io

# Import SPAI from spai/ directory
import sys
sys.path.insert(0, 'spai')
from spai.config import get_config
from spai.models import build_cls_model
from spai.utils import load_pretrained
from spai.data.data_finetune import build_transform


class SPAIDetector:
    """
    SPAI spectral deepfake detector.

    Replaces ELA/FFT forensics with state-of-the-art spectral learning.
    """

    def __init__(
        self,
        config_path: str = "spai/configs/spai.yaml",
        weights_path: str = "spai/weights/spai.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize SPAI model."""
        # Load config
        self.config = get_config({
            "cfg": config_path,
            "batch_size": 1,
            "opts": []
        })

        # Build and load model
        self.model = build_cls_model(self.config)
        self.device = torch.device(device)
        self.model.to(self.device)

        # Load weights
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"SPAI weights not found at {weights_path}. "
                "Download from: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view"
            )

        load_pretrained(self.config, self.model, checkpoint_path=weights_path)
        self.model.eval()

        # Build transform
        self.transform = build_transform(self.config, is_train=False)

    def analyze(
        self,
        image_bytes: bytes,
        generate_heatmap: bool = True,
        alpha: float = 0.6
    ) -> Dict:
        """
        Run SPAI analysis.

        Args:
            image_bytes: Input image as bytes
            generate_heatmap: If True, generate blended attention overlay
            alpha: Transparency for blending (0.6 = 60% original, 40% heatmap)

        Returns:
            {
                "spai_score": float,         # 0.0-1.0 (AI probability)
                "spai_prediction": str,      # "Real" or "AI Generated"
                "spai_confidence": float,    # 0.0-1.0
                "tier": str,                 # "Authentic" / "Suspicious" / "Deepfake"
                "heatmap_bytes": bytes,      # Blended overlay PNG (if requested)
                "analysis_text": str         # Human-readable report
            }
        """
        # Load image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(tensor)
            score = torch.sigmoid(output).item()  # 0-1 probability

        # Determine prediction
        prediction = "AI Generated" if score >= 0.5 else "Real"
        confidence = score if score >= 0.5 else (1.0 - score)

        # Map to tier
        if score >= 0.9:
            tier = "Deepfake"
        elif score >= 0.5:
            tier = "Suspicious"
        else:
            tier = "Authentic"

        # Generate heatmap if requested
        heatmap_bytes = None
        if generate_heatmap:
            heatmap_bytes = self._generate_blended_heatmap(
                tensor,
                pil_image,
                alpha
            )

        # Format analysis text
        analysis_text = self._format_analysis(score, prediction, confidence, tier)

        return {
            "spai_score": score,
            "spai_prediction": prediction,
            "spai_confidence": confidence,
            "tier": tier,
            "heatmap_bytes": heatmap_bytes,
            "analysis_text": analysis_text
        }

    def _generate_blended_heatmap(
        self,
        tensor: torch.Tensor,
        original_image: Image.Image,
        alpha: float
    ) -> bytes:
        """
        Generate blended attention overlay.

        Uses cv2.addWeighted to blend original with heatmap (same as spai/app.py).

        Args:
            tensor: Model input tensor
            original_image: Original PIL Image
            alpha: Transparency (0.6 = 60% original, 40% heatmap)

        Returns:
            PNG bytes of blended overlay
        """
        # Get attention maps from model
        # Note: This may need adjustment based on actual SPAI model API
        with torch.no_grad():
            attention_maps = self.model.get_attention_maps(tensor)

        # Aggregate across layers
        aggregated = attention_maps.mean(dim=0)

        # Resize to original size
        heatmap = torch.nn.functional.interpolate(
            aggregated.unsqueeze(0).unsqueeze(0),
            size=(original_image.height, original_image.width),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # Convert to numpy and normalize
        heatmap_np = heatmap.cpu().numpy()
        heatmap_np = (heatmap_np * 255).clip(0, 255).astype(np.uint8)

        # Apply colormap (JET: red = suspicious, blue = normal)
        heatmap_colored = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

        # Convert BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Convert original to numpy
        original_np = np.array(original_image)

        # Resize heatmap if dimensions don't match
        if heatmap_colored.shape[:2] != original_np.shape[:2]:
            heatmap_colored = cv2.resize(
                heatmap_colored,
                (original_np.shape[1], original_np.shape[0])
            )

        # Blend using cv2.addWeighted (matches spai/app.py)
        beta = 1.0 - alpha
        blended = cv2.addWeighted(original_np, alpha, heatmap_colored, beta, 0)

        # Convert to BGR for PNG encoding
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        # Encode as PNG
        success, png_bytes = cv2.imencode('.png', blended_bgr)

        if not success:
            raise RuntimeError("Failed to encode heatmap as PNG")

        return png_bytes.tobytes()

    def _format_analysis(
        self,
        score: float,
        prediction: str,
        confidence: float,
        tier: str
    ) -> str:
        """Format SPAI analysis as human-readable text."""
        analysis = f"""--- SPAI SPECTRAL ANALYSIS ---

Prediction: {prediction}
SPAI Score: {score:.3f} (0.0=Real, 1.0=AI-Generated)
Confidence: {confidence:.1%}
Risk Tier: {tier}

Spectral Analysis Summary:
The image's frequency spectrum was analyzed using masked feature modeling with
Vision Transformer architecture (CVPR2025). The spectral reconstruction similarity
score indicates {self._get_likelihood_text(score)}.

"""

        if score >= 0.9:
            analysis += "🚨 HIGH CONFIDENCE AI-GENERATED: Strong spectral artifacts detected.\n"
        elif score >= 0.7:
            analysis += "⚠️ LIKELY AI-GENERATED: Moderate spectral inconsistencies found.\n"
        elif score >= 0.5:
            analysis += "⚠️ SUSPICIOUS: Subtle spectral anomalies present.\n"
        elif score >= 0.3:
            analysis += "✓ LIKELY REAL: Spectral patterns consistent with natural images.\n"
        else:
            analysis += "✅ HIGH CONFIDENCE REAL: Natural spectral distribution detected.\n"

        analysis += "\nNote: SPAI analyzes frequency-domain patterns invisible to human perception."

        return analysis

    def _get_likelihood_text(self, score: float) -> str:
        """Convert score to likelihood description."""
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

### Phase 2: Update detector.py

```python
class OSINTDetector:
    """
    SPAI-based deepfake detector with optional VLM assistance.

    Modes:
        - "spai_standalone": SPAI only (fast, no VLM)
        - "spai_assisted": SPAI + VLM (comprehensive)
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
        detection_mode: str = "spai_assisted"
    ):
        """
        Initialize detector.

        Args:
            detection_mode: "spai_standalone" or "spai_assisted"
        """
        self.model_name = model_name
        self.context = context
        self.watermark_mode = watermark_mode
        self.provider = provider
        self.detection_mode = detection_mode

        # Load prompts
        self.prompts = self._load_prompts(prompts_path)

        # Always initialize SPAI
        from spai_detector import SPAIDetector
        self.spai = SPAIDetector()

        # Initialize VLM only if needed
        if detection_mode == "spai_assisted":
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

    def detect(
        self,
        image_bytes: bytes,
        debug: bool = False,
        send_forensics: bool = True  # Renamed to send_spai for clarity
    ) -> Dict:
        """
        Perform deepfake detection.

        Args:
            image_bytes: Image to analyze
            debug: If True, include debug information
            send_forensics: If True in spai_assisted mode, send SPAI analysis to VLM

        Returns:
            {
                "tier": str,
                "confidence": float,
                "reasoning": str,
                "forensic_report": str,  # Now contains SPAI analysis
                "metadata_auto_fail": bool,
                "raw_logits": dict,
                "verdict_token": str,
                "spai_score": float,
                "heatmap_bytes": bytes,
                "detection_mode": str
            }
        """
        # Stage 0: Metadata extraction
        metadata_dict, metadata_report, auto_fail = self._check_metadata(image_bytes)

        if auto_fail:
            return {
                "tier": "Deepfake",
                "confidence": 1.0,
                "reasoning": "AI generation tool detected in image metadata",
                "forensic_report": metadata_report,
                "metadata_auto_fail": True,
                "raw_logits": {"real": -100.0, "fake": 0.0},
                "verdict_token": "B",
                "detection_mode": self.detection_mode
            }

        # Route to appropriate mode
        if self.detection_mode == "spai_standalone":
            return self._detect_spai_standalone(image_bytes, metadata_report)
        else:  # spai_assisted
            return self._detect_spai_assisted(image_bytes, metadata_report, send_forensics)

    def _detect_spai_standalone(
        self,
        image_bytes: bytes,
        metadata_report: str
    ) -> Dict:
        """SPAI-only detection."""
        # Run SPAI
        spai_result = self.spai.analyze(image_bytes, generate_heatmap=True)

        # Determine verdict token
        verdict_token = "B" if spai_result["spai_score"] >= 0.5 else "A"

        # Combine reports
        forensic_report = f"{metadata_report}\n\n{spai_result['analysis_text']}"

        return {
            "tier": spai_result["tier"],
            "confidence": spai_result["spai_score"],
            "reasoning": f"SPAI Standalone Detection\n\n{spai_result['analysis_text']}",
            "forensic_report": forensic_report,
            "metadata_auto_fail": False,
            "raw_logits": {
                "real": np.log(1 - spai_result["spai_score"]) if spai_result["spai_score"] < 1.0 else -100.0,
                "fake": np.log(spai_result["spai_score"]) if spai_result["spai_score"] > 0.0 else -100.0
            },
            "verdict_token": verdict_token,
            "spai_score": spai_result["spai_score"],
            "heatmap_bytes": spai_result["heatmap_bytes"],
            "detection_mode": "spai_standalone"
        }

    def _detect_spai_assisted(
        self,
        image_bytes: bytes,
        metadata_report: str,
        send_spai: bool
    ) -> Dict:
        """SPAI + VLM detection."""
        # Run SPAI analysis
        spai_result = self.spai.analyze(image_bytes, generate_heatmap=True)

        # Build report
        forensic_report = f"{metadata_report}\n\n{spai_result['analysis_text']}"

        # Get system prompt
        system_prompt = self._get_system_prompt(include_forensic_context=send_spai)

        # Run VLM analysis with SPAI assistance
        analysis, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens = \
            self._two_stage_classification_with_spai(
                image_bytes,
                spai_result["heatmap_bytes"],
                forensic_report,
                spai_result["spai_score"],
                system_prompt,
                send_spai
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
            "detection_mode": "spai_assisted",
            "req1_time": req1_time,
            "req2_time": req2_time,
            "req1_tokens": req1_tokens,
            "req2_tokens": req2_tokens
        }

    def _two_stage_classification_with_spai(
        self,
        original_bytes: bytes,
        heatmap_bytes: bytes,
        forensic_report: str,
        spai_score: float,
        system_prompt: str,
        send_spai: bool
    ) -> Tuple[str, Dict, float, float, int, int]:
        """
        Two-stage VLM with SPAI assistance.

        VLM receives:
        - Original image
        - SPAI blended attention overlay
        - SPAI analysis text
        - SPAI score
        """
        original_uri = f"data:image/png;base64,{base64.b64encode(original_bytes).decode()}"

        user_content = []

        if send_spai:
            # Include SPAI analysis
            heatmap_uri = f"data:image/png;base64,{base64.b64encode(heatmap_bytes).decode()}"

            user_content.extend([
                {"type": "text", "text": "--- SPAI SPECTRAL ANALYSIS ---"},
                {"type": "text", "text": forensic_report},
                {"type": "text", "text": f"\nSPAI Score: {spai_score:.3f} (0=Real, 1=AI-Generated)\n"},
                {"type": "text", "text": "--- ORIGINAL IMAGE ---"},
                {"type": "image_url", "image_url": {"url": original_uri}},
                {"type": "text", "text": "--- SPAI ATTENTION OVERLAY (60% Original + 40% Heatmap) ---"},
                {"type": "image_url", "image_url": {"url": heatmap_uri}},
                {
                    "type": "text",
                    "text": (
                        "--- ANALYSIS INSTRUCTIONS ---\n"
                        "You are provided with:\n\n"
                        "1. **SPAI Spectral Analysis**: Frequency-domain analysis from CVPR2025 model\n"
                        f"   - SPAI Score: {spai_score:.3f} ({'AI-generated' if spai_score >= 0.5 else 'Real'})\n\n"
                        "2. **SPAI Attention Overlay**: Semi-transparent heatmap blended onto original\n"
                        "   - Red regions = spectral anomalies detected by SPAI\n"
                        "   - Blue regions = normal spectral patterns\n"
                        "   - Overlay uses 60% original + 40% heatmap transparency\n\n"
                        f"{self.prompts['analysis_instructions']['visual_analysis']}\n\n"
                        "3. **Cross-Reference with SPAI**:\n"
                        "   - Do your visual findings align with SPAI's spectral analysis?\n"
                        "   - Are attention hotspots (red areas) correlated with visual artifacts?\n"
                        "   - Does semantic coherence support or contradict the spectral evidence?\n\n"
                        f"4. **Watermark Analysis**: {self._get_watermark_instruction()}\n\n"
                        "Provide comprehensive reasoning combining visual analysis and spectral findings."
                    )
                }
            ])
        else:
            # Simplified: just original image
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

        # Request 1: Analysis
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
            req1_tokens = len(forensic_report) // 4 + 765 * 2 + 50
        except Exception as e:
            req1_time = time.time() - req1_start
            analysis_text = f"VLM analysis failed: {type(e).__name__}: {str(e)}"
            req1_tokens = 0

        # Request 2: Verdict
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
                "error": f"Verdict extraction failed: {type(e).__name__}: {str(e)}"
            }
            req2_tokens = 0

        return analysis_text, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens
```

### Phase 3: Update app.py

```python
# In batch evaluation section
st.subheader("Detection Mode")
detection_mode = st.radio(
    "Select detection mode",
    options=["spai_standalone", "spai_assisted"],
    format_func=lambda x: {
        "spai_standalone": "🚀 SPAI Standalone (Fast, no VLM)",
        "spai_assisted": "🔬 SPAI-Assisted VLM (Comprehensive)"
    }[x],
    index=1,  # Default to spai_assisted
    help="""
    - SPAI Standalone: Fast spectral analysis only (~50ms/image)
    - SPAI-Assisted VLM: Spectral + semantic analysis (~3s/image)
    """
)

# Initialize detector
detector = OSINTDetector(
    base_url=model_config['base_url'],
    model_name=model_config['model_name'],
    api_key=model_config['api_key'],
    context=eval_context,
    watermark_mode=watermark_mode,
    provider=model_config['provider'],
    detection_mode=detection_mode
)
```

### Phase 4: Delete forensics.py

```bash
git rm forensics.py
```

Update all imports:
- Remove `from forensics import ArtifactGenerator`
- Remove references to `generate_ela`, `generate_fft`, etc.

## Summary

### What's Removed
- ❌ `forensics.py` (645 lines)
- ❌ ELA generation and variance calculation
- ❌ FFT generation and peak detection
- ❌ Forensic report building with ELA/FFT interpretation
- ❌ `vlm_only` mode

### What's Added
- ✅ `spai_detector.py` (~300 lines)
- ✅ SPAI spectral analysis
- ✅ Blended attention overlay (60% original + 40% heatmap)
- ✅ Two simple modes: standalone and assisted

### Architecture Simplification

**Before:**
```
3 modes × 2 forensic types (ELA/FFT) × multiple code paths = Complex
```

**After:**
```
2 modes × 1 forensic type (SPAI) × clean code paths = Simple
```

This is a **clean break** - simpler, faster, more maintainable.
