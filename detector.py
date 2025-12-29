"""
Two-stage OSINT-aware deepfake detection with SPAI spectral analysis.

This module coordinates SPAI spectral analysis, context injection, and verdict extraction
using a hybrid architecture that combines frequency-domain learning with semantic reasoning.
"""

import math
import time
import base64
import yaml
import io
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import Counter
from PIL import Image
from openai import OpenAI
from spai_detector import SPAIDetector
from cloud_providers import get_cloud_adapter
from deepfake_detector import EnhancedDeepfakeDetector


class OSINTDetector:
    """
    SPAI-powered OSINT-specialized deepfake detector.

    Detection Modes:
        - spai_standalone: SPAI spectral analysis only (fast, ~50ms)
        - spai_assisted: SPAI + VLM comprehensive analysis (~3s)
        - enhanced_3layer: Physics + Texture + VLM (comprehensive forensics, ~10-30s)

    Architecture (spai_standalone):
        Stage 1: SPAI spectral analysis
        Stage 2: Three-tier classification

    Architecture (spai_assisted):
        Stage 0: Metadata extraction with auto-fail check
        Stage 1: SPAI spectral analysis with heatmap generation
        Stage 2: VLM analysis with SPAI context (Request 1)
        Stage 3: Verdict extraction with logprobs (Request 2, KV-cached)
        Stage 4: Three-tier classification

    Architecture (enhanced_3layer):
        Layer 1: Physics-Based Forensics (eye reflection, lighting consistency)
        Layer 2: Texture & Artifact Analysis (PLGF, frequency, PLADA)
        Layer 3: VLM semantic analysis
        Final: Weighted voting with confidence calibration

    OSINT Contexts:
        - Military: Uniforms, parades, formations
        - Disaster: Floods, rubble, combat, BDA
        - Propaganda: Studio shots, news, state media
        - Auto: Automatic context detection
    """

    # Token lists for verdict parsing (MCQ format)
    REAL_TOKENS = ['A', ' A', 'a', ' a']
    FAKE_TOKENS = ['B', ' B', 'b', ' b']

    # Metadata blacklist for auto-fail
    AI_TOOL_SIGNATURES = [
        'midjourney', 'stable diffusion', 'stablediffusion',
        'dall-e', 'dalle', 'comfyui', 'automatic1111',
        'invokeai', 'fooocus', 'sora', 'firefly', 'leonardo',
        'nightcafe', 'artbreeder', 'starryai'
    ]

    # Unified tier classification thresholds (applied across ALL detection modes)
    # More conservative than previous thresholds to reduce false positives
    # Symmetric around 0.5 for balanced classification
    TIER_THRESHOLD_DEEPFAKE = 0.75  # P(AI Generated) >= 0.75 → Deepfake
    TIER_THRESHOLD_SUSPICIOUS_LOW = 0.50  # P(AI) > 0.50 → Suspicious (raised from 0.35 to reduce false positives)
    # P(AI Generated) <= 0.50 → Authentic (50%+ confidence in "Real" verdict)

    @classmethod
    def _load_prompts(cls, prompts_path: str = "prompts/current.yaml") -> dict:
        """
        Load prompt templates from YAML configuration file.

        Now uses file-per-version system: loads from prompts/current.yaml
        which points to the active version file.

        Returns:
            dict: Prompts dictionary with metadata section for version tracking
        """
        yaml_path = Path(prompts_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)

        # Log prompt version if metadata exists
        if 'metadata' in prompts:
            metadata = prompts['metadata']
            print(f"📋 Loaded prompts version {metadata.get('version', 'unknown')} "
                  f"(updated: {metadata.get('last_updated', 'unknown')})")

        return prompts

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "dummy",
        context: str = "auto",
        watermark_mode: str = "ignore",
        provider: str = "vllm",
        prompts_path: str = "prompts/current.yaml",
        detection_mode: str = "spai_assisted",
        spai_detector: Optional['SPAIDetector'] = None,
        spai_config_path: str = "spai/configs/spai.yaml",
        spai_weights_path: str = "spai/weights/spai.pth",
        spai_max_size: int = 1280,
        spai_overlay_alpha: float = 0.6
    ):
        """
        Initialize OSINT detector with SPAI integration.

        Args:
            base_url: OpenAI-compatible API endpoint (for vLLM/OpenAI)
            model_name: Model identifier
            api_key: API key (default: "dummy" for vLLM)
            context: "auto", "military", "disaster", or "propaganda"
            watermark_mode: "ignore" (treat as news logos) or "analyze" (flag AI watermarks)
            provider: "vllm", "openai", "anthropic", or "gemini"
            prompts_path: Path to YAML prompts configuration file
            detection_mode: "spai_standalone" (SPAI only), "spai_assisted" (SPAI + VLM), or "enhanced_3layer" (Physics + Texture + VLM)
            spai_detector: Pre-loaded SPAIDetector instance (for caching). If None, creates new instance.
            spai_config_path: Path to SPAI config YAML (only used if spai_detector is None)
            spai_weights_path: Path to SPAI pre-trained weights (only used if spai_detector is None)
            spai_max_size: Maximum resolution for SPAI analysis (512-2048 or None for original)
            spai_overlay_alpha: Transparency for heatmap blending (0.0-1.0, default 0.6)
        """
        self.model_name = model_name
        self.context = context
        self.watermark_mode = watermark_mode
        self.provider = provider
        self.detection_mode = detection_mode
        self.spai_max_size = spai_max_size if spai_max_size != "Original" else None
        self.spai_overlay_alpha = spai_overlay_alpha

        # Provider capability flags
        # Logprobs only available for vLLM and OpenAI, not Gemini Developer API or Anthropic
        self.supports_logprobs = provider in ["vllm", "openai"]

        # Use pre-loaded SPAI detector if provided, otherwise create new one
        if spai_detector is not None:
            self.spai = spai_detector
        else:
            # Fallback: load SPAI detector (slow - 2+ minutes)
            self.spai = SPAIDetector(
                config_path=spai_config_path,
                weights_path=spai_weights_path
            )

        # Load prompts from YAML
        self.prompts = self._load_prompts(prompts_path)

        # Initialize VLM client only if needed for spai_assisted or enhanced_3layer mode
        if detection_mode in ["spai_assisted", "enhanced_3layer"]:
            if provider in ["vllm", "openai"]:
                # Use OpenAI SDK directly (supports both vLLM and OpenAI)
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    timeout=180.0,  # 3 minute timeout to accommodate slower VLM inference
                    max_retries=0  # No retries to fail fast
                )
            else:
                # Use cloud adapter for Anthropic/Gemini
                self.client = get_cloud_adapter(provider, model_name, api_key, base_url)

            # Initialize Enhanced 3-Layer Detector if in enhanced mode
            if detection_mode == "enhanced_3layer":
                # Get model key from model_name for config lookup
                model_key = self._get_model_key_from_name(model_name, base_url)
                self.enhanced_detector = EnhancedDeepfakeDetector(
                    model_key=model_key,
                    enable_all_layers=True
                )
            else:
                self.enhanced_detector = None
        else:
            # Standalone mode doesn't need VLM client
            self.client = None
            self.enhanced_detector = None

    def _get_model_key_from_name(self, model_name: str, base_url: str) -> str:
        """
        Determine model key from model_name and base_url for config lookup.
        Fallback to a generic key if not found.
        """
        # Try to match against known configs
        from config import MODEL_CONFIGS
        for key, conf in MODEL_CONFIGS.items():
            if conf.get("model_name") == model_name or conf.get("base_url") == base_url:
                return key
        # Fallback: use model_name as key
        return model_name

    def detect(
        self,
        image_bytes: bytes,
        debug: bool = False
    ) -> Dict:
        """
        Perform SPAI-powered deepfake detection.

        Args:
            image_bytes: Original image as PNG/JPEG bytes
            debug: If True, include detailed debug information

        Returns:
            {
                "tier": str ("Authentic" / "Suspicious" / "Deepfake"),
                "confidence": float (0.0-1.0),
                "reasoning": str (analysis text),
                "spai_report": str (SPAI spectral analysis report),
                "metadata_auto_fail": bool,
                "raw_logits": {"real": float, "fake": float},
                "verdict_token": str,

                # Debug info (only if debug=True)
                "debug": {...}
            }
        """
        if self.detection_mode == "spai_standalone":
            return self._detect_spai_standalone(image_bytes, debug)
        elif self.detection_mode == "gapl_standalone":
            return self._detect_gapl_standalone(image_bytes, debug)
        elif self.detection_mode == "spai_assisted":
            return self._detect_spai_assisted(image_bytes, debug)
        elif self.detection_mode == "enhanced_3layer":
            return self._detect_enhanced_3layer(image_bytes, debug)
        else:
            raise ValueError(f"Invalid detection_mode: {self.detection_mode}")

    def _detect_spai_standalone(self, image_bytes: bytes, debug: bool = False) -> Dict:
        """
        SPAI standalone mode: Fast spectral analysis without VLM.

        Returns results in ~50ms with SPAI spectral classification only.

        Args:
            image_bytes: Original image bytes
            debug: If True, include debug information

        Returns:
            Detection result dict with SPAI analysis
        """
        pipeline_start = time.time()

        try:
            # Run SPAI analysis with heatmap for visualization
            spai_result = self.spai.analyze(
                image_bytes,
                generate_heatmap=True,  # Generate heatmap for UI display
                alpha=self.spai_overlay_alpha,
                max_size=self.spai_max_size
            )

            # Apply calibration to reduce saturation on domain-mismatched data
            raw_spai_score = spai_result["spai_score"]  # 0.0-1.0 (AI probability)
            calibrated_score = self._calibrate_spai_score(raw_spai_score, temperature=1.5)
            p_fake = calibrated_score  # Use calibrated score for confidence

            # Recalculate tier based on calibrated score (not raw SPAI tier)
            # Uses unified thresholds for consistency across all modes
            if calibrated_score >= self.TIER_THRESHOLD_DEEPFAKE:
                tier = "Deepfake"
            elif calibrated_score > self.TIER_THRESHOLD_SUSPICIOUS_LOW:
                tier = "Suspicious"
            else:
                tier = "Authentic"

            # Create pseudo-logits using calibrated score
            if calibrated_score >= 0.5:
                # AI Generated
                logit_fake = 0.0
                logit_real = -math.log((1 - calibrated_score) / calibrated_score) if calibrated_score < 1.0 else -100.0
            else:
                # Real
                logit_real = 0.0
                logit_fake = -math.log(calibrated_score / (1 - calibrated_score)) if calibrated_score > 0.0 else -100.0

            total_time = time.time() - pipeline_start

            # Add calibration info to reasoning
            enhanced_reasoning = f"{spai_result['analysis_text']}\n\n" \
                                f"**Calibration Applied**: Raw SPAI score {raw_spai_score:.4f} → " \
                                f"Calibrated score {calibrated_score:.4f} (T=1.5) to reduce saturation on HADR/military scenes."

            result = {
                "tier": tier,  # Use recalculated tier from calibrated score
                "confidence": p_fake,  # Return calibrated P(fake)
                "reasoning": enhanced_reasoning,
                "spai_report": enhanced_reasoning,
                "spai_heatmap_bytes": spai_result["heatmap_bytes"],  # Store heatmap for UI display
                "metadata_auto_fail": False,
                "raw_logits": {"real": logit_real, "fake": logit_fake},
                "verdict_token": "B" if tier == "Deepfake" else "A"
            }

            if debug:
                result["debug"] = {
                    "detection_mode": "spai_standalone",
                    "raw_spai_score": raw_spai_score,
                    "calibrated_spai_score": calibrated_score,
                    "spai_prediction": spai_result["spai_prediction"],
                    "spai_tier_original": spai_result["tier"],
                    "tier_after_calibration": tier,
                    "total_pipeline_time": total_time,
                    "spai_inference_time": total_time  # Majority of time is SPAI
                }

            return result

        except Exception as e:
            # Error handling
            return {
                "tier": "Suspicious",
                "confidence": 0.5,
                "reasoning": f"SPAI analysis error: {type(e).__name__}: {str(e)}",
                "spai_report": f"Error during SPAI inference: {str(e)}",
                "metadata_auto_fail": False,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "verdict_token": None
            }

    def _detect_gapl_standalone(self, image_bytes: bytes, debug: bool = False) -> Dict:
        """
        GAPL standalone mode: Generator-aware prototype learning analysis only.

        Returns results with GAPL prototype-based classification (~50-200ms).

        Args:
            image_bytes: Original image bytes
            debug: If True, include debug information

        Returns:
            Detection result dict with GAPL analysis
        """
        pipeline_start = time.time()

        try:
            # Initialize GAPL detector
            from gapl_forensics import GAPLForensicsPipeline
            gapl = GAPLForensicsPipeline(device=self.spai.device)  # Use same device as SPAI

            # Load image from bytes
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_bytes))

            # Run GAPL analysis
            gapl_result = gapl.analyze(image)

            # Map GAPL verdict to tier
            combined_verdict = gapl_result["combined_verdict"]  # "Real", "AI Generated", or "Uncertain"
            confidence_level = gapl_result["confidence"]  # "high", "medium", "low"

            # Map to numeric confidence score
            if combined_verdict == "AI Generated":
                if confidence_level == "high":
                    p_fake = 0.85
                elif confidence_level == "medium":
                    p_fake = 0.65
                else:  # low
                    p_fake = 0.55
            elif combined_verdict == "Real":
                if confidence_level == "high":
                    p_fake = 0.15
                elif confidence_level == "medium":
                    p_fake = 0.35
                else:  # low
                    p_fake = 0.45
            else:  # Uncertain
                p_fake = 0.5

            # Map to tier
            if p_fake >= self.TIER_THRESHOLD_DEEPFAKE:
                tier = "Deepfake"
            elif p_fake > self.TIER_THRESHOLD_SUSPICIOUS_LOW:
                tier = "Suspicious"
            else:
                tier = "Authentic"

            # Create pseudo-logits
            if p_fake >= 0.5:
                logit_fake = 0.0
                logit_real = -math.log((1 - p_fake) / p_fake) if p_fake < 1.0 else -100.0
            else:
                logit_real = 0.0
                logit_fake = -math.log(p_fake / (1 - p_fake)) if p_fake > 0.0 else -100.0

            total_time = time.time() - pipeline_start

            result = {
                "tier": tier,
                "confidence": p_fake,
                "reasoning": gapl_result["explanation"],
                "gapl_report": gapl_result["explanation"],
                "metadata_auto_fail": False,
                "raw_logits": {"real": logit_real, "fake": logit_fake},
                "verdict_token": "B" if tier == "Deepfake" else "A"
            }

            if debug:
                result["debug"] = {
                    "detection_mode": "gapl_standalone",
                    "gapl_verdict": combined_verdict,
                    "gapl_confidence": confidence_level,
                    "gapl_calibrated_score": p_fake,
                    "tier": tier,
                    "total_pipeline_time": total_time,
                    "gapl_inference_time": total_time
                }

            return result

        except Exception as e:
            # Error handling - GAPL may not be available
            return {
                "tier": "Suspicious",
                "confidence": 0.5,
                "reasoning": f"GAPL analysis error: {type(e).__name__}: {str(e)}. GAPL may not be installed or model weights may be missing.",
                "gapl_report": f"Error during GAPL inference: {str(e)}",
                "metadata_auto_fail": False,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "verdict_token": None
            }

    def _detect_spai_assisted(self, image_bytes: bytes, debug: bool = False) -> Dict:
        """
        SPAI assisted mode: SPAI spectral analysis + VLM comprehensive reasoning.

        Uses SPAI heatmap overlay as visual input to VLM for enhanced analysis.

        Args:
            image_bytes: Original image bytes
            debug: If True, include debug information

        Returns:
            Detection result dict with SPAI + VLM analysis
        """
        pipeline_start = time.time()

        # Stage 0: Check metadata for auto-fail
        try:
            stage0_start = time.time()
            metadata_dict, metadata_report, auto_fail = self._check_metadata(image_bytes)
            stage0_time = time.time() - stage0_start
        except Exception as e:
            # If even metadata extraction fails, return minimal error result
            return {
                "tier": "Suspicious",
                "confidence": 0.5,
                "reasoning": f"Pipeline initialization error: {type(e).__name__}: {str(e)}",
                "spai_report": "Error: Could not extract metadata",
                "metadata_auto_fail": False,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "verdict_token": None
            }

        if auto_fail:
            result = {
                "tier": "Deepfake",
                "confidence": 1.0,
                "reasoning": "AI generation tool detected in image metadata",
                "spai_report": metadata_report,
                "metadata_auto_fail": True,
                "raw_logits": {"real": -100.0, "fake": 0.0},
                "verdict_token": "B"
            }

            if debug:
                result["debug"] = {
                    "detection_mode": "spai_assisted",
                    "system_prompt": "N/A (auto-fail)",
                    "exif_data": metadata_dict,
                    "context_applied": self.context,
                    "request_1_latency": 0.0,
                    "request_2_latency": 0.0,
                    "request_1_tokens": 0,
                    "request_2_tokens": 0,
                    "top_k_logprobs": [],
                    "kv_cache_hit": False,
                    "total_pipeline_time": time.time() - pipeline_start
                }

            return result

        # Stage 1: Run SPAI analysis with heatmap generation (with error handling)
        try:
            stage1_start = time.time()
            spai_result = self.spai.analyze(
                image_bytes,
                generate_heatmap=True,  # Generate blended overlay for VLM
                alpha=self.spai_overlay_alpha,
                max_size=self.spai_max_size
            )
            stage1_time = time.time() - stage1_start

            # Apply calibration to SPAI score
            raw_spai_score = spai_result["spai_score"]
            calibrated_spai_score = self._calibrate_spai_score(raw_spai_score, temperature=1.5)

            # Build SPAI report combining metadata + spectral analysis + calibration
            spai_report = f"""{metadata_report}

{spai_result['analysis_text']}

SPAI Scores:
- Raw spectral score: {raw_spai_score:.4f}
- Calibrated score (T=1.5): {calibrated_spai_score:.4f}
- Prediction: {spai_result['spai_prediction']}

Note: Calibration applied to reduce saturation on domain-mismatched HADR/military scenes.

OSINT Context: {self.context.capitalize()}
"""
        except Exception as e:
            # If SPAI generation fails, return error with metadata at least
            return {
                "tier": "Suspicious",
                "confidence": 0.5,
                "reasoning": f"SPAI analysis error: {type(e).__name__}: {str(e)}",
                "spai_report": f"{metadata_report}\n\nError: Could not generate SPAI analysis",
                "metadata_auto_fail": False,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "verdict_token": None
            }

        # Stage 2 & 3: Two-stage VLM calls with SPAI context (with error handling)
        vlm_available = True  # Track if VLM actually ran successfully
        try:
            system_prompt = self._get_system_prompt_spai()
            analysis, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens = self._two_stage_classification_spai(
                image_bytes,
                spai_result["heatmap_bytes"],
                spai_report,
                system_prompt
            )
            # Check if VLM actually failed (not just logprobs unavailable)
            # VLM is considered unavailable only if we got an error AND no valid token
            if "error" in verdict_result and verdict_result.get("token") is None:
                vlm_available = False
                analysis = f"VLM Service Unavailable: {verdict_result.get('error', 'Unknown error')}"
            # If we have a valid token (even without logprobs), VLM succeeded
            elif verdict_result.get("token") is None:
                # No token and no error - something unexpected happened
                vlm_available = False
                analysis = "VLM Service Unavailable: No verdict token returned"
        except Exception as e:
            # If VLM calls fail, fall back to SPAI-only mode
            vlm_available = False
            analysis = f"VLM analysis error: {type(e).__name__}: {str(e)}\n\nFalling back to SPAI-only analysis."
            verdict_result = {
                "token": None,
                "confidence": calibrated_spai_score if 'calibrated_spai_score' in locals() else 0.5,
                "raw_logits": {"real": None, "fake": None},
                "error": str(e)
            }
            req1_time = req2_time = req1_tokens = req2_tokens = 0

        # Stage 4: Disagreement gating (prevent error amplification)
        # Only apply if VLM is available
        raw_spai_score = spai_result["spai_score"]
        calibrated_spai = self._calibrate_spai_score(raw_spai_score, temperature=1.5)

        if vlm_available:
            spai_says_fake = calibrated_spai > 0.7  # SPAI thinks fake (after calibration)
            vlm_confidence = verdict_result["confidence"]

            # Only perform disagreement detection if VLM confidence is numeric
            if isinstance(vlm_confidence, (int, float)):
                vlm_says_fake = vlm_confidence > 0.7  # VLM thinks fake

                # Strong disagreement: one says fake with confidence > 0.7, other disagrees
                strong_disagreement = (spai_says_fake and not vlm_says_fake) or (not spai_says_fake and vlm_says_fake)
            else:
                # VLM confidence not available (string "not available") - skip disagreement detection
                strong_disagreement = False

            if strong_disagreement:
                # Override to Suspicious with averaged confidence (only if both are numeric)
                if isinstance(vlm_confidence, (int, float)):
                    avg_confidence = (calibrated_spai + vlm_confidence) / 2.0
                else:
                    avg_confidence = calibrated_spai  # Use SPAI only if VLM confidence unavailable

                tier = "Suspicious"
                verdict_result["confidence"] = avg_confidence
                verdict_result["disagreement_detected"] = True

                # Append disagreement note to reasoning
                vlm_conf_display = f"{vlm_confidence:.3f}" if isinstance(vlm_confidence, (int, float)) else str(vlm_confidence)
                analysis += f"\n\n**DISAGREEMENT DETECTED**: SPAI (spectral) and VLM (semantic) produced conflicting verdicts. " \
                            f"SPAI calibrated score: {calibrated_spai:.3f}, VLM confidence: {vlm_conf_display}. " \
                            f"Marking as Suspicious due to model disagreement - manual review recommended."
            else:
                verdict_result["disagreement_detected"] = False

                # Original tier classification
                tier = self._classify_tier(verdict_result["confidence"])

                # If confidence is not available, use token-based classification
                # Trust the VLM's binary decision (A=Authentic, B=Deepfake)
                if tier == "not available" and verdict_result["token"]:
                    if verdict_result["token"] in self.FAKE_TOKENS:
                        tier = "Deepfake"  # VLM says fake → trust it
                    elif verdict_result["token"] in self.REAL_TOKENS:
                        tier = "Authentic"  # VLM says real → trust it
                    else:
                        tier = "Suspicious"  # No clear token → uncertain

                    # Add note about logprobs unavailability (Gemini/Claude)
                    if not self.supports_logprobs:
                        if isinstance(verdict_result['confidence'], (int, float)):
                            # Successfully extracted confidence from text
                            analysis += f"\n\n**Note**: Using {self.provider.capitalize()} provider which does not support log probabilities. " \
                                       f"VLM confidence ({verdict_result['confidence']:.2f}) extracted from text-based response rather than token logprobs."
                        else:
                            # Failed to extract confidence - using token only
                            analysis += f"\n\n**Note**: Using {self.provider.capitalize()} provider which does not support log probabilities. " \
                                       f"Classification based on VLM verdict token ({verdict_result['token']}) without calibrated confidence scores."
                elif tier == "not available":
                    tier = "Suspicious"  # No verdict at all → conservative
        else:
            # VLM unavailable - fall back to SPAI-only classification using unified thresholds
            verdict_result["disagreement_detected"] = False
            if calibrated_spai >= self.TIER_THRESHOLD_DEEPFAKE:
                tier = "Deepfake"
            elif calibrated_spai > self.TIER_THRESHOLD_SUSPICIOUS_LOW:
                tier = "Suspicious"
            else:
                tier = "Authentic"
            verdict_result["confidence"] = calibrated_spai
            analysis += f"\n\n**SPAI-ONLY MODE**: VLM unavailable, using calibrated SPAI score ({calibrated_spai:.3f}) for classification."

        total_time = time.time() - pipeline_start

        result = {
            "tier": tier,
            "confidence": verdict_result["confidence"],
            "reasoning": analysis,
            "spai_report": spai_report,
            "spai_heatmap_bytes": spai_result["heatmap_bytes"],  # Store heatmap for UI display
            "metadata_auto_fail": False,
            "raw_logits": verdict_result.get("raw_logits", {"real": None, "fake": None}),
            "verdict_token": verdict_result.get("token"),
            "vlm_available": vlm_available  # Track VLM availability for UI
        }

        if debug:
            result["debug"] = {
                "detection_mode": "spai_assisted",
                "system_prompt": system_prompt,
                "exif_data": metadata_dict,
                "raw_spai_score": raw_spai_score,
                "calibrated_spai_score": calibrated_spai,
                "spai_prediction": spai_result["spai_prediction"],
                "spai_tier": spai_result["tier"],
                "disagreement_detected": verdict_result.get("disagreement_detected", False),
                "context_applied": self.context,
                "threshold_adjustments": self._get_threshold_adjustments(),
                "request_1_latency": req1_time,
                "request_2_latency": req2_time,
                "request_1_tokens": req1_tokens,
                "request_2_tokens": req2_tokens,
                "top_k_logprobs": verdict_result.get("top_k_logprobs", []),
                "kv_cache_hit": req2_time < 0.5,  # Heuristic: <0.5s likely cached
                "stage_0_time": stage0_time,
                "stage_1_time": stage1_time,
                "total_pipeline_time": total_time
            }

        return result

    def _detect_enhanced_3layer(self, image_bytes: bytes, debug: bool = False) -> Dict:
        """
        Enhanced 4-layer mode: Texture + GAPL + SPAI + VLM with weighted voting.

        This mode runs the EXACT same SPAI+VLM analysis as spai_assisted mode, but adds:
        - Layer 1: Texture forensics (PLGF, frequency, PLADA) - appended to SPAI report
        - Layer 2: GAPL (Generator-Aware Prototype Learning) - appended to SPAI report
        - Layer 3: SPAI spectral analysis with calibration
        - Layer 4: VLM semantic reasoning (sees all forensics context)

        Final verdict determined by weighted voting across all 4 layers.

        Args:
            image_bytes: Original image bytes
            debug: If True, include debug information

        Returns:
            Detection result dict (same format as spai_assisted + layer details)
        """
        pipeline_start = time.time()

        # Import forensics modules
        from texture_forensics import TextureForensicsPipeline
        from gapl_forensics import GAPLForensicsPipeline

        # Stage 0: Check metadata for auto-fail (SAME as spai_assisted)
        try:
            stage0_start = time.time()
            metadata_dict, metadata_report, auto_fail = self._check_metadata(image_bytes)
            stage0_time = time.time() - stage0_start
        except Exception as e:
            return {
                "tier": "Suspicious",
                "confidence": 0.5,
                "reasoning": f"Pipeline initialization error: {type(e).__name__}: {str(e)}",
                "spai_report": "Error: Could not extract metadata",
                "metadata_auto_fail": False,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "verdict_token": None
            }

        if auto_fail:
            result = {
                "tier": "Deepfake",
                "confidence": 1.0,
                "reasoning": "AI generation tool detected in image metadata",
                "spai_report": metadata_report,
                "metadata_auto_fail": True,
                "raw_logits": {"real": -100.0, "fake": 0.0},
                "verdict_token": "B"
            }
            if debug:
                result["debug"] = {
                    "detection_mode": "enhanced_3layer",
                    "exif_data": metadata_dict,
                    "total_pipeline_time": time.time() - pipeline_start
                }
            return result

        # Convert to PIL Image for forensics layers
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Layer 1: Texture & Artifact Analysis
        try:
            layer1_start = time.time()
            texture_pipeline = TextureForensicsPipeline()
            layer1_result = texture_pipeline.analyze(pil_image)
            layer1_time = time.time() - layer1_start
        except Exception as e:
            layer1_result = {
                "combined_verdict": "Error",
                "confidence": "low",
                "explanation": f"Texture layer failed: {str(e)}"
            }
            layer1_time = 0.0

        # Layer 2: GAPL (Generator-Aware Prototype Learning) Analysis
        try:
            layer2_start = time.time()
            gapl_pipeline = GAPLForensicsPipeline()
            layer2_result = gapl_pipeline.analyze(pil_image)
            layer2_time = time.time() - layer2_start
        except Exception as e:
            layer2_result = {
                "combined_verdict": "Uncertain",
                "confidence": "low",
                "explanation": f"GAPL layer failed: {str(e)}"
            }
            layer2_time = 0.0

        # Stage 1: Run SPAI analysis with heatmap (SAME as spai_assisted)
        try:
            stage1_start = time.time()
            spai_result = self.spai.analyze(
                image_bytes,
                generate_heatmap=True,
                alpha=self.spai_overlay_alpha,
                max_size=self.spai_max_size
            )
            stage1_time = time.time() - stage1_start

            # Extract raw SPAI score for calibration and voting
            raw_spai_score = spai_result["spai_score"]
            calibrated_spai_score = self._calibrate_spai_score(raw_spai_score, temperature=1.5)

            # Build ENHANCED SPAI report with Texture & GAPL results APPENDED
            # The VLM will receive ALL forensics data as context
            spai_report = f"""{metadata_report}

{spai_result['analysis_text']}

SPAI Scores:
- Raw spectral score: {raw_spai_score:.4f}
- Calibrated score (T=1.5): {calibrated_spai_score:.4f}
- Prediction: {spai_result['spai_prediction']}

Note: Calibration applied to reduce saturation on domain-mismatched HADR/military scenes.

OSINT Context: {self.context.capitalize()}

--- ADDITIONAL FORENSICS (LAYER 1 & 2) ---

LAYER 1 - TEXTURE ANALYSIS:
{layer1_result.get('explanation', 'No texture analysis available')}
→ Texture Verdict: {layer1_result.get('combined_verdict', 'N/A')} (Confidence: {layer1_result.get('confidence', 'N/A')})

LAYER 2 - GAPL (GENERATOR-AWARE):
{layer2_result.get('explanation', 'No GAPL analysis available')}
→ GAPL Verdict: {layer2_result.get('combined_verdict', 'N/A')} (Confidence: {layer2_result.get('confidence', 'N/A')})
"""
        except Exception as e:
            return {
                "tier": "Suspicious",
                "confidence": 0.5,
                "reasoning": f"SPAI analysis error: {type(e).__name__}: {str(e)}",
                "spai_report": f"{metadata_report}\n\nError: Could not generate SPAI analysis",
                "metadata_auto_fail": False,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "verdict_token": None
            }

        # Stage 2 & 3: Two-stage VLM calls with enhanced SPAI report (SAME as spai_assisted)
        # The VLM now receives forensics data + SPAI data and analyzes everything together
        vlm_available = True  # Track if VLM actually ran successfully
        try:
            system_prompt = self._get_system_prompt_spai()
            analysis, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens = self._two_stage_classification_spai(
                image_bytes,
                spai_result["heatmap_bytes"],
                spai_report,  # <-- Enhanced with Texture & GAPL forensics!
                system_prompt
            )
            # Check if VLM actually failed (not just logprobs unavailable)
            # VLM is considered unavailable only if we got an error AND no valid token
            if "error" in verdict_result and verdict_result.get("token") is None:
                vlm_available = False
                analysis = f"VLM Service Unavailable: {verdict_result.get('error', 'Unknown error')}"
            # If we have a valid token (even without logprobs), VLM succeeded
            elif verdict_result.get("token") is None:
                # No token and no error - something unexpected happened
                vlm_available = False
                analysis = "VLM Service Unavailable: No verdict token returned"
        except Exception as e:
            vlm_available = False
            analysis = f"VLM analysis error: {type(e).__name__}: {str(e)}"
            verdict_result = {
                "token": None,
                "confidence": None,
                "raw_logits": {"real": None, "fake": None},
                "error": str(e)
            }
            req1_time = req2_time = req1_tokens = req2_tokens = 0

        # Stage 4: WEIGHTED VOTING across all 4 layers (Texture + GAPL + SPAI + VLM)

        # Prepare SPAI verdict (already calibrated) using unified thresholds
        if calibrated_spai_score >= self.TIER_THRESHOLD_DEEPFAKE:
            spai_tier = "Deepfake"
        elif calibrated_spai_score > self.TIER_THRESHOLD_SUSPICIOUS_LOW:
            spai_tier = "Suspicious"
        else:
            spai_tier = "Authentic"

        spai_verdict_dict = {
            "tier": spai_tier,
            "confidence": calibrated_spai_score
        }

        # Prepare VLM verdict (only if VLM is available)
        if vlm_available:
            vlm_tier = self._classify_tier(verdict_result["confidence"])
            if vlm_tier == "not available" and verdict_result["token"]:
                if verdict_result["token"] in self.FAKE_TOKENS:
                    vlm_tier = "Deepfake"
                elif verdict_result["token"] in self.REAL_TOKENS:
                    vlm_tier = "Authentic"
                else:
                    vlm_tier = "Suspicious"

                # Add note about logprobs unavailability (Gemini/Claude)
                if not self.supports_logprobs:
                    if isinstance(verdict_result['confidence'], (int, float)):
                        # Successfully extracted confidence from text
                        analysis += f"\n\n**Note**: Using {self.provider.capitalize()} provider which does not support log probabilities. " \
                                   f"VLM confidence ({verdict_result['confidence']:.2f}) extracted from text-based response rather than token logprobs."
                    else:
                        # Failed to extract confidence - using token only
                        analysis += f"\n\n**Note**: Using {self.provider.capitalize()} provider which does not support log probabilities. " \
                                   f"VLM classification based on verdict token ({verdict_result['token']}) without calibrated confidence scores."
            elif vlm_tier == "not available":
                vlm_tier = "Suspicious"

            vlm_verdict_dict = {
                "tier": vlm_tier,
                "confidence": verdict_result["confidence"]
            }
        else:
            # VLM unavailable - will be excluded from voting
            vlm_verdict_dict = None
            vlm_tier = "N/A"

        # Perform weighted voting
        final_tier, final_confidence, consensus = self._weighted_voting(
            layer1_texture=layer1_result,
            layer2_gapl=layer2_result,
            spai_verdict=spai_verdict_dict,
            vlm_verdict=vlm_verdict_dict  # Will be None if VLM unavailable
        )

        total_time = time.time() - pipeline_start

        # Build voting explanation
        # Format VLM result based on availability
        if vlm_available and verdict_result.get('confidence') is not None:
            # Check if confidence is numeric before formatting
            vlm_conf = verdict_result['confidence']
            if isinstance(vlm_conf, (int, float)):
                vlm_display = f"{vlm_tier} ({vlm_conf:.3f})"
            else:
                vlm_display = f"{vlm_tier} ({vlm_conf})"  # String "not available"
        else:
            vlm_display = "N/A (Service Down)"

        # Format final_confidence safely (should always be float from _weighted_voting, but be defensive)
        if isinstance(final_confidence, (int, float)):
            final_conf_display = f"{final_confidence:.3f}"
        else:
            final_conf_display = str(final_confidence)

        voting_explanation = f"\n\n**WEIGHTED VOTING RESULT**:\n" \
                            f"- Texture: {layer1_result.get('combined_verdict', 'N/A')} ({layer1_result.get('confidence', 'N/A')})\n" \
                            f"- GAPL: {layer2_result.get('combined_verdict', 'N/A')} ({layer2_result.get('confidence', 'N/A')})\n" \
                            f"- SPAI: {spai_tier} ({calibrated_spai_score:.3f})\n" \
                            f"- VLM: {vlm_display}\n" \
                            f"- **Final Verdict: {final_tier}** (confidence: {final_conf_display}, consensus: {consensus})"

        # Return result with layer details for UI
        result = {
            "tier": final_tier,  # Use weighted voting result
            "confidence": final_confidence,  # Use weighted confidence
            "reasoning": analysis + voting_explanation,  # Append voting explanation
            "spai_report": spai_report,  # Enhanced report with all layers
            "spai_heatmap_bytes": spai_result["heatmap_bytes"],
            "metadata_auto_fail": False,
            "raw_logits": verdict_result.get("raw_logits", {"real": None, "fake": None}),
            "verdict_token": verdict_result.get("token"),
            "vlm_available": vlm_available,  # Track VLM availability for UI
            # Store layer results for UI display
            "layer1_texture": layer1_result,
            "layer2_gapl": layer2_result,
            "layer_agreement": {
                "texture": layer1_result.get("combined_verdict", "N/A"),
                "gapl": layer2_result.get("combined_verdict", "N/A"),
                "spai": spai_tier,
                "vllm": vlm_tier,
                "final_weighted": final_tier,
                "consensus": consensus
            }
        }

        if debug:
            result["debug"] = {
                "detection_mode": "enhanced_3layer",
                "system_prompt": system_prompt,
                "exif_data": metadata_dict,
                "raw_spai_score": raw_spai_score,
                "calibrated_spai_score": calibrated_spai_score,
                "spai_prediction": spai_result["spai_prediction"],
                "spai_tier_original": spai_result["tier"],
                "spai_tier_calibrated": spai_tier,
                "vlm_tier": vlm_tier,
                "layer1_texture": layer1_result,
                "layer2_gapl": layer2_result,
                "weighted_voting_result": {
                    "final_tier": final_tier,
                    "final_confidence": final_confidence,
                    "consensus": consensus
                },
                "layer1_time": layer1_time,
                "layer2_time": layer2_time,
                "context_applied": self.context,
                "request_1_latency": req1_time,
                "request_2_latency": req2_time,
                "request_1_tokens": req1_tokens,
                "request_2_tokens": req2_tokens,
                "top_k_logprobs": verdict_result.get("top_k_logprobs", []),
                "stage_0_time": stage0_time,
                "stage_1_time": stage1_time,
                "total_pipeline_time": total_time
            }

        return result

    def _weighted_voting(
        self,
        layer1_texture: Dict,
        layer2_gapl: Dict,
        spai_verdict: Dict,
        vlm_verdict: Dict
    ) -> Tuple[str, float, bool]:
        """
        Perform weighted voting across all detection layers.

        Args:
            layer1_texture: Texture forensics result dict
            layer2_gapl: GAPL (Generator-Aware Prototype Learning) forensics result dict
            spai_verdict: SPAI spectral verdict {"tier": str, "confidence": float}
            vlm_verdict: VLM semantic verdict {"tier": str, "confidence": float} or None if unavailable

        Returns:
            (combined_tier, combined_confidence, consensus_flag)
        """
        # Collect verdicts with confidence weights
        verdicts = {}
        confidences = {}

        # Texture layer
        if layer1_texture and layer1_texture.get("combined_verdict") not in ["Error", "Inconclusive", "Uncertain"]:
            verdicts["texture"] = layer1_texture["combined_verdict"]
            confidences["texture"] = layer1_texture.get("confidence", "low")

        # GAPL layer
        if layer2_gapl and layer2_gapl.get("combined_verdict") not in ["Error", "Inconclusive", "Uncertain"]:
            verdicts["gapl"] = layer2_gapl["combined_verdict"]
            confidences["gapl"] = layer2_gapl.get("confidence", "low")

        # SPAI layer (convert tier to verdict format)
        # NOTE: Now includes "Suspicious" verdicts - they participate in voting
        if spai_verdict and spai_verdict.get("tier"):
            spai_tier = spai_verdict["tier"]
            spai_conf = spai_verdict.get("confidence", 0.5)

            # Only process if confidence is numeric (should always be for SPAI)
            if isinstance(spai_conf, (int, float)):
                # Map tier to verdict format based on confidence
                # "Suspicious" verdicts now vote based on confidence threshold
                if spai_tier == "Deepfake":
                    verdicts["spai"] = "AI Generated"
                elif spai_tier == "Authentic":
                    verdicts["spai"] = "Real"
                elif spai_tier == "Suspicious":
                    # Suspicious votes based on confidence: >= 0.55 votes "AI Generated", < 0.45 votes "Real"
                    if spai_conf >= 0.55:
                        verdicts["spai"] = "AI Generated"
                    elif spai_conf < 0.45:
                        verdicts["spai"] = "Real"
                    # Between 0.45-0.55: Don't vote (too uncertain)

                # Convert confidence to categorical if verdict was added
                if "spai" in verdicts:
                    if spai_conf >= 0.8:
                        confidences["spai"] = "high"
                    elif spai_conf >= 0.5:
                        confidences["spai"] = "medium"
                    else:
                        confidences["spai"] = "low"

        # VLM layer (convert tier to verdict format)
        # NOTE: Now includes "Suspicious" verdicts - they participate in voting
        # IMPORTANT: Only include VLM if it's not None (service was available)
        if vlm_verdict is not None and vlm_verdict.get("tier"):
            vlm_tier = vlm_verdict["tier"]
            vlm_conf = vlm_verdict.get("confidence", 0.5)

            # Check if confidence is numeric (may be "not available" string for Gemini/Claude)
            if isinstance(vlm_conf, (int, float)):
                # Map tier to verdict format based on confidence
                # "Suspicious" verdicts now vote based on confidence threshold
                if vlm_tier == "Deepfake":
                    verdicts["vllm"] = "AI Generated"
                elif vlm_tier == "Authentic":
                    verdicts["vllm"] = "Real"
                elif vlm_tier == "Suspicious":
                    # Suspicious votes based on confidence: >= 0.55 votes "AI Generated", < 0.45 votes "Real"
                    if vlm_conf >= 0.55:
                        verdicts["vllm"] = "AI Generated"
                    elif vlm_conf < 0.45:
                        verdicts["vllm"] = "Real"
                    # Between 0.45-0.55: Don't vote (too uncertain)

                # Convert confidence to categorical if verdict was added
                if "vllm" in verdicts:
                    if vlm_conf >= 0.8:
                        confidences["vllm"] = "high"
                    elif vlm_conf >= 0.5:
                        confidences["vllm"] = "medium"
                    else:
                        confidences["vllm"] = "low"
            else:
                # VLM confidence not available (string) - use tier only for voting
                if vlm_tier == "Deepfake":
                    verdicts["vllm"] = "AI Generated"
                    confidences["vllm"] = "medium"  # Assume medium confidence when numeric unavailable
                elif vlm_tier == "Authentic":
                    verdicts["vllm"] = "Real"
                    confidences["vllm"] = "medium"
                # Skip "Suspicious" tier when no numeric confidence available

        # No valid verdicts → Suspicious
        if not verdicts:
            return "Suspicious", 0.5, False

        # Check if VLM is corroborated by any forensic/spectral layer
        forensic_verdicts = {k: v for k, v in verdicts.items() if k in ["texture", "gapl", "spai"]}
        vllm_corroborated = False

        if "vllm" in verdicts:
            vllm_verdict_str = verdicts["vllm"]
            for fv in forensic_verdicts.values():
                # Exact match or semantic alignment (both non-Real)
                if fv == vllm_verdict_str or (fv != "Real" and vllm_verdict_str != "Real"):
                    vllm_corroborated = True
                    break

        # Weighted voting
        confidence_weights = {"high": 3, "medium": 2, "low": 1}
        weighted_votes = []

        for layer, verdict in verdicts.items():
            # Safe access to confidence with default "low" if missing
            layer_confidence = confidences.get(layer, "low")
            weight = confidence_weights.get(layer_confidence, 1)

            # Cap VLM weight if uncorroborated by forensics/SPAI
            if layer == "vllm" and not vllm_corroborated:
                weight = min(weight, 1)  # Cap at 1 vote

            weighted_votes.extend([verdict] * weight)

        # Count votes
        verdict_counts = Counter(weighted_votes)

        # Special handling for "AI Manipulated" (only texture layer can detect this)
        if "texture" in verdicts and verdicts["texture"] == "AI Manipulated":
            texture_conf = confidences.get("texture", "low")
            if texture_conf in ["high", "medium"]:
                manipulated_votes = verdict_counts.get("AI Manipulated", 0)
                ai_gen_votes = verdict_counts.get("AI Generated", 0)

                # Prefer "AI Manipulated" if it has reasonable support
                if manipulated_votes > 0 and manipulated_votes >= ai_gen_votes * 0.4:
                    combined_verdict = "AI Manipulated"
                else:
                    combined_verdict = verdict_counts.most_common(1)[0][0]
            else:
                combined_verdict = verdict_counts.most_common(1)[0][0]
        else:
            # Normal majority vote
            combined_verdict = verdict_counts.most_common(1)[0][0]

        # Check consensus among all verdicts that participated in voting
        # Note: Error/Uncertain/Inconclusive were already filtered out when building verdicts dict
        unique_verdicts = set(verdicts.values())
        consensus = len(unique_verdicts) == 1  # True only if ALL voting layers agree

        # Convert verdict back to tier format
        if combined_verdict == "AI Generated":
            final_tier = "Deepfake"
        elif combined_verdict == "AI Manipulated":
            final_tier = "Suspicious"  # Manipulated = uncertain for now
        elif combined_verdict == "Real":
            final_tier = "Authentic"
        else:
            final_tier = "Suspicious"

        # Calculate combined confidence (average of agreeing layers)
        agreeing_layers = [layer for layer, v in verdicts.items() if v == combined_verdict]
        if agreeing_layers:
            # Map categorical to numeric
            conf_map = {"high": 0.9, "medium": 0.65, "low": 0.4}
            avg_conf = sum(conf_map.get(confidences[layer], 0.5) for layer in agreeing_layers) / len(agreeing_layers)
        else:
            avg_conf = 0.5

        # Convert avg_conf to P(AI Generated) for consistent interpretation
        # avg_conf represents "confidence in the voted verdict"
        # We need to convert it to "P(AI Generated)" for display
        if final_tier == "Authentic":
            # Layers voted "Real" with avg_conf confidence
            # So P(AI Generated) = 1 - avg_conf
            p_ai_generated = 1.0 - avg_conf
        else:
            # Layers voted "Deepfake/Suspicious" with avg_conf confidence
            # So P(AI Generated) = avg_conf
            p_ai_generated = avg_conf

        # RE-CLASSIFY tier based on P(AI Generated) to ensure consistency
        # This prevents contradictions like "Authentic (AI Generated: 77.5%)"
        # Uses unified thresholds for consistency across all modes
        if p_ai_generated >= self.TIER_THRESHOLD_DEEPFAKE:
            final_tier = "Deepfake"
        elif p_ai_generated > self.TIER_THRESHOLD_SUSPICIOUS_LOW:
            final_tier = "Suspicious"
        else:
            final_tier = "Authentic"

        # Return P(AI Generated) as the confidence value
        return final_tier, p_ai_generated, consensus

    def _calibrate_spai_score(self, raw_score: float, temperature: float = 1.5) -> float:
        """
        Apply temperature scaling to SPAI score to reduce saturation.

        Problem: SPAI outputs near-0 or near-1 scores on domain-mismatched data
        (HADR scenes, night shots, smoke/fire, compression), making thresholding ineffective.

        Args:
            raw_score: Raw SPAI probability (0-1)
            temperature: T > 1 reduces confidence (default 1.5 for balanced calibration)

        Returns:
            Calibrated score (0-1)
        """
        # Convert probability to logit
        if raw_score >= 0.9999:
            logit = 9.0  # Cap extreme values
        elif raw_score <= 0.0001:
            logit = -9.0
        else:
            logit = math.log(raw_score / (1.0 - raw_score))

        # Apply temperature scaling
        scaled_logit = logit / temperature

        # Convert back to probability
        calibrated_score = 1.0 / (1.0 + math.exp(-scaled_logit))

        return calibrated_score

    def _extract_metadata_simple(self, image_bytes: bytes) -> Dict[str, str]:
        """
        Extract EXIF metadata from image bytes.

        Args:
            image_bytes: Image data as bytes

        Returns:
            Dictionary of EXIF tag names to values (empty if none found)
        """
        try:
            import exifread
            import io

            file_obj = io.BytesIO(image_bytes)
            tags = exifread.process_file(file_obj, details=False)

            # Convert to simple string dict
            metadata_dict = {}
            for tag_name, tag_value in tags.items():
                # Skip thumbnail data
                if 'thumbnail' in tag_name.lower():
                    continue
                metadata_dict[tag_name] = str(tag_value)

            return metadata_dict
        except Exception:
            # If metadata extraction fails, return empty dict
            return {}

    def _check_metadata(self, image_bytes: bytes) -> Tuple[Dict, str, bool]:
        """
        Extract EXIF metadata and check for AI tool signatures.

        Returns:
            (metadata_dict, metadata_report, auto_fail)
        """
        metadata_dict = self._extract_metadata_simple(image_bytes)

        # Build metadata report
        report_lines = ["EXIF Metadata:"]
        auto_fail = False

        if not metadata_dict:
            report_lines.append("  (No EXIF data found)")
        else:
            for tag_name, tag_value in metadata_dict.items():
                report_lines.append(f"  {tag_name}: {tag_value}")

                # Check for AI tool signatures
                tag_lower = str(tag_value).lower()
                for signature in self.AI_TOOL_SIGNATURES:
                    if signature in tag_lower:
                        auto_fail = True
                        report_lines.append(f"  ⚠️ AI TOOL DETECTED: {signature}")
                        break

        if not auto_fail:
            report_lines.append("  ✓ Auto-Fail Check: PASSED (no AI tools detected)")

        return metadata_dict, "\n".join(report_lines), auto_fail

    def _two_stage_classification_spai(
        self,
        original_bytes: bytes,
        heatmap_bytes: bytes,
        spai_report: str,
        system_prompt: str
    ) -> Tuple[str, Dict, float, float, int, int]:
        """
        Perform two-stage API calls with SPAI spectral analysis context.

        Args:
            original_bytes: Original image bytes
            heatmap_bytes: SPAI blended attention heatmap bytes (60% original + 40% heatmap)
            spai_report: SPAI spectral analysis report text
            system_prompt: System prompt for VLM

        Returns:
            (analysis_text, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens)
        """
        # Convert images to base64
        original_uri = f"data:image/png;base64,{base64.b64encode(original_bytes).decode()}"
        heatmap_uri = f"data:image/png;base64,{base64.b64encode(heatmap_bytes).decode()}"

        # Build user message content with SPAI as TEXT metadata (NO HEATMAP IMAGE - prevents anchoring)
        user_content = [
            {"type": "text", "text": "--- SPAI SPECTRAL ANALYSIS (REFERENCE DATA) ---"},
            {"type": "text", "text": spai_report},
            {"type": "text", "text": "\n**IMPORTANT**: SPAI is a frequency-domain model optimized for pristine images. "
                                     "It may misclassify authentic HADR/military scenes (night shots, smoke, debris, heavy compression) as fake. "
                                     "Use SPAI as ONE data point, but prioritize your own visual analysis of semantic coherence, "
                                     "lighting physics, and context-appropriate artifacts.\n"},
            {"type": "text", "text": "--- IMAGE TO ANALYZE ---"},
            {"type": "image_url", "image_url": {"url": original_uri}},
            {
                "type": "text",
                "text": (
                    f"{self.prompts['analysis_instructions']['spai_instructions']}\n"
                    "--- ANALYSIS INSTRUCTIONS ---\n"
                    "Analyze this image independently. Use SPAI's spectral data as reference, "
                    "but form your own conclusion based on visual semantics.\n\n"
                    f"{self.prompts['analysis_instructions']['visual_analysis']}\n\n"
                    "2. **SPAI Spectral Reference**:\n"
                    "   - SPAI detected frequency patterns (see report above)\n"
                    "   - Consider SPAI's analysis, but DO NOT over-anchor on it\n"
                    "   - SPAI struggles with: night scenes, smoke/fire, debris, heavy compression\n"
                    "   - If SPAI says 'fake' but visuals look semantically coherent, explain the discrepancy\n\n"
                    f"{self.prompts['analysis_instructions']['metadata_instructions']}\n\n"
                    f"4. **Watermark Analysis**: {self._get_watermark_instruction()}\n\n"
                    "Provide your comprehensive reasoning. If you disagree with SPAI, explain why."
                )
            }
        ]

        # Request 1: Analysis
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content
            }
        ]

        req1_start = time.time()
        try:
            # Use adapter's create_completion for cloud providers, or direct OpenAI SDK
            if hasattr(self.client, 'create_completion'):
                response_1 = self.client.create_completion(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=4000
                )
            else:
                response_1 = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=4000
                )
            req1_time = time.time() - req1_start

            analysis_text = response_1.choices[0].message.content

            # Estimate tokens (rough approximation: 1 token ≈ 4 chars for text, 765 per image)
            req1_tokens = len(spai_report) // 4 + 765 * 2 + 50  # 2 images (original + heatmap) + text overhead

        except Exception as e:
            # Fallback on API error with descriptive message
            error_detail = f"Stage 2 (VLM Analysis) failed: {type(e).__name__}: {str(e)}"
            return (
                error_detail,
                {
                    "token": None,
                    "confidence": 0.5,
                    "raw_logits": {"real": -0.69, "fake": -0.69},
                    "error": str(e)
                },
                0.0, 0.0, 0, 0
            )

        # Append assistant response to history
        messages.append({"role": "assistant", "content": analysis_text})

        # Request 2: Verdict (KV-cache optimized)
        # Load verdict prompt from prompts.yaml - use Gemini-specific prompt if provider is Gemini/Claude
        if self.provider in ["gemini", "anthropic"]:
            verdict_prompt = self.prompts.get('verdict_prompt_gemini',
                self.prompts.get('verdict_prompt',
                    """Based on your analysis, provide your final verdict:

(A) Real (Authentic Capture)
(B) Fake (AI Generated/Manipulated)

Answer with ONLY the single letter A or B."""
                )
            ).strip()
        else:
            verdict_prompt = self.prompts.get('verdict_prompt',
                """Based on your analysis, provide your final verdict:

(A) Real (Authentic Capture)
(B) Fake (AI Generated/Manipulated)

Answer with ONLY the single letter A or B."""
            ).strip()

        messages.append({
            "role": "user",
            "content": verdict_prompt
        })

        req2_start = time.time()
        try:
            # Use adapter's create_completion for cloud providers, or direct OpenAI SDK
            # Only request logprobs if provider supports them (vLLM/OpenAI)
            if hasattr(self.client, 'create_completion'):
                if self.supports_logprobs:
                    response_2 = self.client.create_completion(
                        messages=messages,
                        temperature=0.0,
                        max_tokens=1000,
                        logprobs=True,
                        top_logprobs=5
                    )
                else:
                    response_2 = self.client.create_completion(
                        messages=messages,
                        temperature=0.0,
                        max_tokens=1000
                    )
            else:
                if self.supports_logprobs:
                    response_2 = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=1000,
                        logprobs=True,
                        top_logprobs=5
                    )
                else:
                    response_2 = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=1000
                    )
            req2_time = time.time() - req2_start

            # Parse verdict (with or without logprobs depending on provider)
            verdict_result = self._parse_verdict(response_2, logprobs_requested=self.supports_logprobs)

            # Request 2 tokens = Request 1 output + verdict prompt (KV-cache reuses system + images)
            # Estimate: 1 token ≈ 4 chars for text
            req2_tokens = len(analysis_text) // 4 + len(verdict_prompt) // 4

        except Exception as e:
            req2_time = time.time() - req2_start
            verdict_result = {
                "token": None,
                "confidence": 0.5,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "error": f"Stage 3 (Verdict) failed: {type(e).__name__}: {str(e)}",
                "top_k_logprobs": []
            }
            req2_tokens = 0

        return analysis_text, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens

    def _get_system_prompt_spai(self) -> str:
        """
        Generate SPAI-aware OSINT system prompt from YAML configuration.

        Uses SPAI protocols instead of forensic protocols.
        """
        # Base prompt from YAML
        base = self.prompts['system_prompts']['base']

        # Get SPAI protocols from YAML
        case_a = self.prompts['system_prompts']['spai_protocols']['case_a']
        case_b = self.prompts['system_prompts']['spai_protocols']['case_b']
        case_c = self.prompts['system_prompts']['spai_protocols']['case_c']

        if self.context == "auto":
            protocols = f"{case_a}\n{case_b}\n{case_c}"
        elif self.context == "military":
            protocols = case_a
        elif self.context == "disaster":
            protocols = case_b
        elif self.context == "propaganda":
            protocols = case_c
        else:
            # Default to auto if unknown context
            protocols = f"{case_a}\n{case_b}\n{case_c}"

        return f"{base}\n{protocols}"

    def _parse_verdict(self, response, logprobs_requested: bool = True) -> Dict:
        """
        Parse verdict from API response (with or without logprobs).

        Args:
            response: API response object
            logprobs_requested: Whether logprobs were requested in the API call

        Returns:
            {
                "token": str,
                "confidence": float or "not available",
                "raw_logits": {"real": float, "fake": float} or {"real": "N/A", "fake": "N/A"},
                "top_k_logprobs": List[Tuple[str, float]]  # For debug mode
            }
        """
        # If logprobs weren't requested (Gemini/Claude), skip directly to text parsing
        if not logprobs_requested:
            return self._parse_verdict_from_text(response, expected=True)

        # Try to parse logprobs (for vLLM/OpenAI)
        try:
            # Check if logprobs structure exists before accessing
            if (response.choices[0].logprobs is None or
                not hasattr(response.choices[0].logprobs, 'content') or
                response.choices[0].logprobs.content is None or
                len(response.choices[0].logprobs.content) == 0):
                # Logprobs not available - fall back to text parsing
                return self._parse_verdict_from_text(response, expected=False)

            logprobs_content = response.choices[0].logprobs.content[0]
            token = logprobs_content.token
            top_logprobs = logprobs_content.top_logprobs

            # Collect all top-k logprobs for debug
            top_k_list = [(lp.token, lp.logprob) for lp in top_logprobs]

            score_real = -100.0
            score_fake = -100.0

            for logprob_obj in top_logprobs:
                t = logprob_obj.token
                lp = logprob_obj.logprob

                if t in self.REAL_TOKENS and score_real == -100.0:
                    score_real = lp
                elif t in self.FAKE_TOKENS and score_fake == -100.0:
                    score_fake = lp

            # Softmax normalization
            p_real = math.exp(score_real)
            p_fake = math.exp(score_fake)
            confidence_fake = p_fake / (p_fake + p_real)

            return {
                "token": token,
                "confidence": confidence_fake,
                "raw_logits": {"real": score_real, "fake": score_fake},
                "top_k_logprobs": top_k_list
            }

        except Exception as e:
            # Unexpected error during logprobs parsing - fall back to text
            return self._parse_verdict_from_text(response, expected=False, error=e)

    def _parse_verdict_from_text(self, response, expected: bool = False, error: Exception = None) -> Dict:
        """
        Parse verdict from text response (fallback when logprobs unavailable).

        Args:
            response: API response object
            expected: True if logprobs weren't requested (Gemini/Claude), False if parsing failed unexpectedly
            error: Original exception if this is a fallback from failed logprobs parsing

        Returns:
            Verdict dict with token and "not available" confidence
        """
        try:
            response_text = response.choices[0].message.content.strip()
            import re

            # First, try to parse Gemini-specific format: CONFIDENCE_SCORE + VERDICT + KEY_FLAW
            gemini_confidence_match = re.search(r'CONFIDENCE_SCORE:\s*(\d{1,3})', response_text, re.IGNORECASE)
            gemini_verdict_match = re.search(r'VERDICT:\s*(REAL|FAKE)', response_text, re.IGNORECASE)

            if gemini_confidence_match and gemini_verdict_match:
                # Gemini format detected!
                conf_pct = int(gemini_confidence_match.group(1))
                verdict_str = gemini_verdict_match.group(1).upper()

                # Convert verdict to token
                if verdict_str == 'FAKE':
                    token = 'B'
                    confidence = conf_pct / 100.0  # Confidence represents P(Fake)
                elif verdict_str == 'REAL':
                    token = 'A'
                    confidence = 1.0 - (conf_pct / 100.0)  # Confidence represents P(Real), so P(Fake) = 1 - conf
                else:
                    token = None
                    confidence = "not available"

                # Try to extract KEY_FLAW for logging
                key_flaw_match = re.search(r'KEY_FLAW:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
                key_flaw = key_flaw_match.group(1).strip() if key_flaw_match else "Not specified"

                result = {
                    "token": token,
                    "confidence": confidence,
                    "raw_logits": {"real": "N/A", "fake": "N/A"},
                    "top_k_logprobs": [],
                    "key_flaw": key_flaw  # Additional Gemini-specific field
                }

                # Only add error field if logprobs were expected but failed
                if not expected and error:
                    result["error"] = f"Logprobs parsing failed: {str(error)}"

                return result

            # Fallback: Look for standalone A or B in the response (original format)
            response_upper = response_text.upper()
            standalone_match = re.search(r'(?:^|[\s\n\r])\(?([AB])\)?(?:[\s\n\r\.]|$)', response_upper)

            if standalone_match:
                token = standalone_match.group(1)
            elif response_upper in ['A', 'B']:
                token = response_upper
            elif '(A)' in response_upper:
                token = 'A'
            elif '(B)' in response_upper:
                token = 'B'
            else:
                # Search anywhere in response
                if 'A' in response_upper and 'B' not in response_upper:
                    token = 'A'
                elif 'B' in response_upper and 'A' not in response_upper:
                    token = 'B'
                else:
                    token = None

            # Try to extract confidence percentage from text (NEW: for Gemini/Claude)
            # Expected format: "A\n75" or "B\n85" (token on first line, confidence on second)
            confidence = "not available"

            # Look for confidence percentage (0-100) on separate line or after token
            confidence_patterns = [
                r'\n\s*(\d{1,3})\s*$',  # Confidence on new line at end: "A\n75"
                r'(?:^|[\n\r])[AB]\s*[\n\r]\s*(\d{1,3})',  # Token, newline, confidence: "A\n75"
                r'(?:confidence|score)[:\s]+(\d{1,3})',  # "confidence: 75"
            ]

            for pattern in confidence_patterns:
                conf_match = re.search(pattern, response_text, re.IGNORECASE)
                if conf_match:
                    try:
                        conf_pct = int(conf_match.group(1))
                        if 0 <= conf_pct <= 100:
                            # Convert percentage to probability
                            # If token is "B" (Fake), confidence represents P(Fake)
                            # If token is "A" (Real), confidence represents P(Real), so P(Fake) = 1 - conf
                            if token == 'B':
                                confidence = conf_pct / 100.0
                            elif token == 'A':
                                confidence = 1.0 - (conf_pct / 100.0)
                            else:
                                # No valid token, use raw probability
                                confidence = conf_pct / 100.0
                            break
                    except (ValueError, IndexError):
                        continue

            result = {
                "token": token,
                "confidence": confidence,
                "raw_logits": {"real": "N/A", "fake": "N/A"},
                "top_k_logprobs": []
            }

            # Only add error field if logprobs were expected but failed
            if not expected and error:
                result["error"] = f"Logprobs parsing failed: {str(error)}"

            return result

        except Exception as text_error:
            # Complete fallback - couldn't even parse text
            return {
                "token": None,
                "confidence": "not available",
                "raw_logits": {"real": "N/A", "fake": "N/A"},
                "top_k_logprobs": [],
                "error": f"Text parsing failed: {str(text_error)}" + (f" (original error: {str(error)})" if error else "")
            }

    def _classify_tier(self, confidence_fake) -> str:
        """
        Classify into standardized three-tier system.

        Args:
            confidence_fake: Either a float (0.0-1.0) or "not available"

        Returns:
            One of: "Deepfake", "Suspicious", or "Authentic"
            (or "not available" if confidence is string - caller handles token-based mapping)

        Confidence-based Thresholds (UNIFIED):
            - confidence_fake >= 0.75 → Deepfake (high confidence AI-generated)
            - 0.50 < confidence_fake < 0.75 → Suspicious (moderate confidence)
            - confidence_fake <= 0.50 → Authentic (likely real)

        Token-based Mapping (when logprobs unavailable):
            - Token "B" (Fake) → Deepfake (trust VLM's binary decision)
            - Token "A" (Real) → Authentic (trust VLM's binary decision)
        """
        # Handle string case (when logprobs not available)
        if isinstance(confidence_fake, str):
            return "not available"  # Will be replaced by token-based classification

        # Normal float-based classification using unified thresholds
        if confidence_fake >= self.TIER_THRESHOLD_DEEPFAKE:
            return "Deepfake"
        elif confidence_fake > self.TIER_THRESHOLD_SUSPICIOUS_LOW:
            return "Suspicious"
        else:
            return "Authentic"

    def _get_watermark_instruction(self) -> str:
        """Get watermark analysis instruction based on mode."""
        if self.watermark_mode == "analyze":
            return self.prompts['watermark_instructions']['analyze']
        else:  # ignore mode (default)
            return self.prompts['watermark_instructions']['ignore']

    def _get_threshold_adjustments(self) -> Dict[str, str]:
        """Get context-specific analysis adjustments for debug display."""
        adjustments = {}

        if self.context == "military":
            adjustments["Formation Patterns"] = "Regular grid patterns expected (not flagged as AI)"
            adjustments["Uniform Consistency"] = "High visual similarity expected"
        elif self.context == "disaster":
            adjustments["Spectral Noise"] = "High-entropy chaos expected (not flagged)"
            adjustments["Damage Textures"] = "Irregular patterns are normal"
        elif self.context == "propaganda":
            adjustments["Post-Processing"] = "Professional editing expected (not flagged as AI)"
            adjustments["Studio Lighting"] = "Controlled environment indicators normal"

        return adjustments
