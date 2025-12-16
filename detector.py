"""
Two-stage OSINT-aware deepfake detection with SPAI spectral analysis.

This module coordinates SPAI spectral analysis, context injection, and verdict extraction
using a hybrid architecture that combines frequency-domain learning with semantic reasoning.
"""

import math
import time
import base64
import yaml
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from openai import OpenAI
from spai_detector import SPAIDetector
from cloud_providers import get_cloud_adapter


class OSINTDetector:
    """
    SPAI-powered OSINT-specialized deepfake detector.

    Detection Modes:
        - spai_standalone: SPAI spectral analysis only (fast, ~50ms)
        - spai_assisted: SPAI + VLM comprehensive analysis (~3s)

    Architecture (spai_standalone):
        Stage 1: SPAI spectral analysis
        Stage 2: Three-tier classification

    Architecture (spai_assisted):
        Stage 0: Metadata extraction with auto-fail check
        Stage 1: SPAI spectral analysis with heatmap generation
        Stage 2: VLM analysis with SPAI context (Request 1)
        Stage 3: Verdict extraction with logprobs (Request 2, KV-cached)
        Stage 4: Three-tier classification

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

    @classmethod
    def _load_prompts(cls, prompts_path: str = "prompts.yaml") -> dict:
        """Load prompt templates from YAML configuration file."""
        yaml_path = Path(prompts_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "dummy",
        context: str = "auto",
        watermark_mode: str = "ignore",
        provider: str = "vllm",
        prompts_path: str = "prompts.yaml",
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
            detection_mode: "spai_standalone" (SPAI only) or "spai_assisted" (SPAI + VLM)
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

        # Initialize VLM client only if needed for spai_assisted mode
        if detection_mode == "spai_assisted":
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
        else:
            # Standalone mode doesn't need VLM client
            self.client = None

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
        elif self.detection_mode == "spai_assisted":
            return self._detect_spai_assisted(image_bytes, debug)
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

            # Convert SPAI confidence to logits for consistency
            spai_score = spai_result["spai_score"]  # 0.0-1.0 (AI probability)
            confidence_fake = spai_result["spai_confidence"]

            # Create pseudo-logits (for consistency with VLM mode)
            if spai_score >= 0.5:
                # AI Generated
                logit_fake = 0.0
                logit_real = -math.log((1 - spai_score) / spai_score) if spai_score < 1.0 else -100.0
            else:
                # Real
                logit_real = 0.0
                logit_fake = -math.log(spai_score / (1 - spai_score)) if spai_score > 0.0 else -100.0

            total_time = time.time() - pipeline_start

            result = {
                "tier": spai_result["tier"],
                "confidence": confidence_fake,
                "reasoning": spai_result["analysis_text"],
                "spai_report": spai_result["analysis_text"],
                "spai_heatmap_bytes": spai_result["heatmap_bytes"],  # Store heatmap for UI display
                "metadata_auto_fail": False,
                "raw_logits": {"real": logit_real, "fake": logit_fake},
                "verdict_token": "B" if spai_result["spai_prediction"] == "AI Generated" else "A"
            }

            if debug:
                result["debug"] = {
                    "detection_mode": "spai_standalone",
                    "spai_score": spai_score,
                    "spai_prediction": spai_result["spai_prediction"],
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

            # Build SPAI report combining metadata + spectral analysis
            spai_report = f"""{metadata_report}

{spai_result['analysis_text']}

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
        try:
            system_prompt = self._get_system_prompt_spai()
            analysis, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens = self._two_stage_classification_spai(
                image_bytes,
                spai_result["heatmap_bytes"],
                spai_report,
                system_prompt
            )
        except Exception as e:
            # If VLM calls fail, return SPAI report with error message
            return {
                "tier": "Suspicious",
                "confidence": 0.5,
                "reasoning": f"VLM analysis error: {type(e).__name__}: {str(e)}",
                "spai_report": spai_report,  # At least show SPAI analysis
                "metadata_auto_fail": False,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "verdict_token": None
            }

        # Stage 4: Determine tier based on confidence
        tier = self._classify_tier(verdict_result["confidence"])

        total_time = time.time() - pipeline_start

        result = {
            "tier": tier,
            "confidence": verdict_result["confidence"],
            "reasoning": analysis,
            "spai_report": spai_report,
            "spai_heatmap_bytes": spai_result["heatmap_bytes"],  # Store heatmap for UI display
            "metadata_auto_fail": False,
            "raw_logits": verdict_result["raw_logits"],
            "verdict_token": verdict_result["token"]
        }

        if debug:
            result["debug"] = {
                "detection_mode": "spai_assisted",
                "system_prompt": system_prompt,
                "exif_data": metadata_dict,
                "spai_score": spai_result["spai_score"],
                "spai_prediction": spai_result["spai_prediction"],
                "spai_tier": spai_result["tier"],
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

        # Build user message content with SPAI context
        user_content = [
            {"type": "text", "text": "--- SPAI SPECTRAL ANALYSIS REPORT ---"},
            {"type": "text", "text": spai_report},
            {"type": "text", "text": "--- ORIGINAL IMAGE ---"},
            {"type": "image_url", "image_url": {"url": original_uri}},
            {"type": "text", "text": "--- SPAI ATTENTION HEATMAP OVERLAY ---"},
            {"type": "text", "text": "This overlay shows SPAI's spectral attention map blended onto the original image (60% original + 40% heatmap). "
                                     "WARM COLORS (red/orange/yellow) indicate HIGHER DEGREE of AI manipulation, with DARK RED showing HIGHEST CONFIDENCE of AI-generated artifacts. "
                                     "COOL COLORS (light blue to dark blue) indicate LOWEST CONFIDENCE of AI manipulation (likely authentic regions)."},
            {"type": "image_url", "image_url": {"url": heatmap_uri}},
            {
                "type": "text",
                "text": (
                    f"{self.prompts['analysis_instructions']['spai_instructions']}\n"
                    "--- ANALYSIS INSTRUCTIONS ---\n"
                    "Analyze this image using the SPAI spectral analysis as context.\n\n"
                    f"{self.prompts['analysis_instructions']['visual_analysis']}\n\n"
                    "2. **SPAI Spectral Correlation**:\n"
                    "   - Review the SPAI analysis report and attention heatmap overlay\n"
                    "   - Use SPAI's frequency-domain insights alongside your visual analysis\n"
                    "   - WARM COLORS (red/orange/yellow) indicate higher AI manipulation confidence, with dark red = highest\n"
                    "   - COOL COLORS (light to dark blue) indicate lower AI manipulation confidence (likely authentic)\n"
                    "   - Pay special attention to warm-colored regions when checking for visual artifacts\n"
                    "   - Apply the appropriate OSINT protocol for this scene type\n\n"
                    f"{self.prompts['analysis_instructions']['metadata_instructions']}\n\n"
                    f"4. **Watermark Analysis**: {self._get_watermark_instruction()}\n\n"
                    "Provide your comprehensive reasoning for whether this image is authentic or AI-generated."
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
            response_1 = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=2000  # Extended to accommodate detailed analysis requirements
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
        verdict_prompt = """Based on your analysis, provide your final verdict:

(A) Real (Authentic Capture)
(B) Fake (AI Generated/Manipulated)

Answer with ONLY the single letter A or B."""

        messages.append({
            "role": "user",
            "content": verdict_prompt
        })

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

            # Parse verdict logprobs
            verdict_result = self._parse_verdict(response_2)

            req2_tokens = len(verdict_prompt.split())

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

    def _parse_verdict(self, response) -> Dict:
        """
        Parse verdict logprobs from API response.

        Returns:
            {
                "token": str,
                "confidence": float,
                "raw_logits": {"real": float, "fake": float},
                "top_k_logprobs": List[Tuple[str, float]]  # For debug mode
            }
        """
        try:
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
            return {
                "token": None,
                "confidence": 0.5,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "top_k_logprobs": [],
                "error": str(e)
            }

    def _classify_tier(self, confidence_fake: float) -> str:
        """
        Classify into three-tier system.

        Thresholds:
            - confidence_fake >= 0.90 → Deepfake
            - 0.50 <= confidence_fake < 0.90 → Suspicious
            - confidence_fake < 0.50 → Authentic
        """
        if confidence_fake >= 0.90:
            return "Deepfake"
        elif confidence_fake >= 0.50:
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
