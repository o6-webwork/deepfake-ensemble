"""
Two-stage OSINT-aware deepfake detection with KV-cache optimization.

This module coordinates forensic analysis, context injection, and verdict extraction
using a hybrid "Cyborg" architecture that combines signal processing with semantic reasoning.
"""

import math
import time
import base64
from typing import Dict, Tuple, List, Optional
from openai import OpenAI
from forensics import ArtifactGenerator


class OSINTDetector:
    """
    Two-stage OSINT-specialized deepfake detector.

    Architecture:
        Stage 0: Metadata extraction with auto-fail check
        Stage 1: Forensic artifact generation (ELA + FFT)
        Stage 2: VLM analysis with reasoning (Request 1)
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

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "dummy",
        context: str = "auto"
    ):
        """
        Initialize OSINT detector.

        Args:
            base_url: OpenAI-compatible API endpoint
            model_name: Model identifier
            api_key: API key (default: "dummy" for vLLM)
            context: "auto", "military", "disaster", or "propaganda"
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.context = context
        self.artifact_gen = ArtifactGenerator()

    def detect(
        self,
        image_bytes: bytes,
        debug: bool = False
    ) -> Dict:
        """
        Perform two-stage OSINT-aware deepfake detection.

        Args:
            image_bytes: Original image as PNG/JPEG bytes
            debug: If True, include detailed debug information

        Returns:
            {
                "tier": str ("Authentic" / "Suspicious" / "Deepfake"),
                "confidence": float (0.0-1.0),
                "reasoning": str (analysis from stage 2),
                "forensic_report": str (ELA/FFT/EXIF findings),
                "metadata_auto_fail": bool,
                "raw_logits": {"real": float, "fake": float},
                "verdict_token": str,

                # Debug info (only if debug=True)
                "debug": {...}
            }
        """
        pipeline_start = time.time()

        # Stage 0: Check metadata for auto-fail
        stage0_start = time.time()
        metadata_dict, metadata_report, auto_fail = self._check_metadata(image_bytes)
        stage0_time = time.time() - stage0_start

        if auto_fail:
            result = {
                "tier": "Deepfake",
                "confidence": 1.0,
                "reasoning": "AI generation tool detected in image metadata",
                "forensic_report": metadata_report,
                "metadata_auto_fail": True,
                "raw_logits": {"real": -100.0, "fake": 0.0},
                "verdict_token": "B"
            }

            if debug:
                result["debug"] = {
                    "system_prompt": "N/A (auto-fail)",
                    "exif_data": metadata_dict,
                    "ela_variance": 0.0,
                    "fft_pattern": "N/A",
                    "fft_peaks": 0,
                    "context_applied": self.context,
                    "threshold_adjustments": {},
                    "request_1_latency": 0.0,
                    "request_2_latency": 0.0,
                    "request_1_tokens": 0,
                    "request_2_tokens": 0,
                    "top_k_logprobs": [],
                    "kv_cache_hit": False,
                    "total_pipeline_time": time.time() - pipeline_start
                }

            return result

        # Stage 1: Generate forensic artifacts
        stage1_start = time.time()
        ela_bytes = self.artifact_gen.generate_ela(image_bytes)
        fft_bytes, fft_metrics = self.artifact_gen.generate_fft_preprocessed(image_bytes)
        ela_variance = self.artifact_gen.compute_ela_variance(ela_bytes)
        stage1_time = time.time() - stage1_start

        # Build forensic report text
        forensic_report = self._build_forensic_report(
            metadata_report,
            ela_variance,
            fft_metrics
        )

        # Stage 2 & 3: Two-stage API calls
        system_prompt = self._get_system_prompt()
        analysis, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens = self._two_stage_classification(
            image_bytes,
            ela_bytes,
            fft_bytes,
            forensic_report,
            system_prompt
        )

        # Stage 4: Determine tier based on confidence
        tier = self._classify_tier(verdict_result["confidence"])

        total_time = time.time() - pipeline_start

        result = {
            "tier": tier,
            "confidence": verdict_result["confidence"],
            "reasoning": analysis,
            "forensic_report": forensic_report,
            "metadata_auto_fail": False,
            "raw_logits": verdict_result["raw_logits"],
            "verdict_token": verdict_result["token"]
        }

        if debug:
            result["debug"] = {
                "system_prompt": system_prompt,
                "exif_data": metadata_dict,
                "ela_variance": ela_variance,
                "fft_pattern": fft_metrics['pattern_type'],
                "fft_peaks": fft_metrics['peaks_detected'],
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

    def _check_metadata(self, image_bytes: bytes) -> Tuple[Dict, str, bool]:
        """
        Extract EXIF metadata and check for AI tool signatures.

        Returns:
            (metadata_dict, metadata_report, auto_fail)
        """
        metadata_dict = self.artifact_gen.extract_metadata(image_bytes)

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

    def _build_forensic_report(
        self,
        metadata_report: str,
        ela_variance: float,
        fft_metrics: Dict
    ) -> str:
        """Generate human-readable forensic report."""

        # ELA interpretation
        if ela_variance < 2.0:
            ela_interp = "Low variance → Uniform compression (AI indicator)"
        else:
            ela_interp = "High variance → Inconsistent compression (manipulation indicator)"

        report = f"""{metadata_report}

ELA Analysis:
  Variance Score: {ela_variance:.2f}
  Threshold: <2.0 (AI indicator)
  Interpretation: {ela_interp}

FFT Analysis:
  Pattern Type: {fft_metrics['pattern_type']}
  Peaks Detected: {fft_metrics['peaks_detected']} above threshold
  Peak Threshold: {fft_metrics['peak_threshold']:.0f}
  Interpretation: {'Grid/Starfield/Cross = AI artifacts' if fft_metrics['pattern_type'] != 'Chaotic' else 'Chaotic = Natural frequencies'}

OSINT Context: {self.context.capitalize()}
"""

        # Add context-specific adjustments if not auto
        if self.context != "auto":
            adjustments = self._get_threshold_adjustments()
            if adjustments:
                report += "\nContext-Adaptive Adjustments:\n"
                for key, value in adjustments.items():
                    report += f"  - {key}: {value}\n"

        return report.strip()

    def _two_stage_classification(
        self,
        original_bytes: bytes,
        ela_bytes: bytes,
        fft_bytes: bytes,
        forensic_report: str,
        system_prompt: str
    ) -> Tuple[str, Dict, float, float, int, int]:
        """
        Perform two-stage API calls for analysis + verdict.

        Returns:
            (analysis_text, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens)
        """
        # Convert images to base64
        original_uri = f"data:image/png;base64,{base64.b64encode(original_bytes).decode()}"
        ela_uri = f"data:image/png;base64,{base64.b64encode(ela_bytes).decode()}"
        fft_uri = f"data:image/png;base64,{base64.b64encode(fft_bytes).decode()}"

        # Request 1: Analysis (with forensic evidence)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "--- FORENSIC LAB REPORT ---"},
                    {"type": "text", "text": forensic_report},
                    {"type": "text", "text": "--- ORIGINAL IMAGE ---"},
                    {"type": "image_url", "image_url": {"url": original_uri}},
                    {"type": "text", "text": "--- ELA MAP ---"},
                    {"type": "image_url", "image_url": {"url": ela_uri}},
                    {"type": "text", "text": "--- FFT SPECTRUM ---"},
                    {"type": "image_url", "image_url": {"url": fft_uri}},
                    {
                        "type": "text",
                        "text": (
                            "Analyze the above forensic evidence. "
                            "Provide your reasoning for whether this image is "
                            "authentic or AI-generated."
                        )
                    }
                ]
            }
        ]

        req1_start = time.time()
        try:
            response_1 = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=300
            )
            req1_time = time.time() - req1_start

            analysis_text = response_1.choices[0].message.content

            # Estimate tokens (rough approximation: 1 token ≈ 4 chars for text, 765 per image)
            req1_tokens = len(forensic_report) // 4 + 765 * 3 + 50  # 3 images + text overhead

        except Exception as e:
            # Fallback on API error
            return (
                f"Error in Stage 2 analysis: {str(e)}",
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
                "error": str(e)
            }
            req2_tokens = 0

        return analysis_text, verdict_result, req1_time, req2_time, req1_tokens, req2_tokens

    def _get_system_prompt(self) -> str:
        """Generate context-adaptive OSINT system prompt."""
        base = """You are a Senior OSINT Image Analyst specializing in military, disaster, and propaganda imagery verification."""

        case_a = """
CASE A: Military Context (Uniforms/Parades/Formations)
- Filter: IGNORE repetitive grid artifacts in FFT (formations create patterns).
- Focus: Look for clone stamp errors (duplicate faces, floating weapons).
- Threshold: FFT peak threshold increased by +20%."""

        case_b = """
CASE B: Disaster/HADR Context (Flood/Rubble/Combat)
- Filter: IGNORE high-entropy noise (chaos is expected).
- Focus: Look for physics failures (liquid fire, smoke without shadows).
- Do NOT flag "messy" textures as suspicious."""

        case_c = """
CASE C: Propaganda/Showcase Context (Studio/News)
- Filter: Expect high ELA contrast (post-processing is normal).
- Focus: Distinguish "beautification" from "generation" artifacts.
- Check metadata for professional camera signatures."""

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

    def _get_threshold_adjustments(self) -> Dict[str, str]:
        """Get context-specific threshold adjustments for debug display."""
        adjustments = {}

        if self.context == "military":
            adjustments["FFT Peak Threshold"] = "+20% (24 instead of 20)"
            adjustments["Grid Artifacts"] = "Ignored (formations expected)"
        elif self.context == "disaster":
            adjustments["High-Entropy Noise"] = "Ignored (chaos expected)"
            adjustments["Messy Textures"] = "NOT flagged as suspicious"
        elif self.context == "propaganda":
            adjustments["ELA Contrast"] = "High values expected (post-processing)"
            adjustments["Beautification Filter"] = "Distinguished from generation"

        return adjustments
