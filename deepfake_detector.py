"""
Enhanced Deepfake Detector with 3-Layer Pipeline

This refactored module integrates:
- Layer 1: Physics-Based Forensics (Eye Reflection + Lighting Consistency)
- Layer 2: Texture & Artifact Analysis (PLGF + Frequency + PLADA)
- Layer 3: Vision-Language Model Analysis

Optimized for social media compressed images.
"""

from PIL import Image
from typing import Dict, Union, List
import numpy as np
import cv2
import io
import base64
from collections import Counter

from physics_forensics import PhysicsForensicsPipeline
from texture_forensics import TextureForensicsPipeline
from config import get_client_and_name


class EnhancedDeepfakeDetector:
    """
    Enhanced deepfake detector with 3-layer forensic analysis.

    This detector combines multiple compression-resistant techniques:
    1. Physics-based forensics (geometry + lighting)
    2. Texture & artifact analysis (PLGF + frequency + PLADA)
    3. VLM-based semantic analysis
    """

    def __init__(self, model_key: str = "MiniCPM-V-4_5", enable_all_layers: bool = True):
        """
        Initialize the enhanced detector.

        Args:
            model_key: Model key for VLM layer (from config.py)
            enable_all_layers: If True, run all 3 layers. If False, only VLM.
        """
        self.model_key = model_key
        self.enable_all_layers = enable_all_layers

        # Initialize forensic pipelines
        if enable_all_layers:
            self.physics_pipeline = PhysicsForensicsPipeline()
            self.texture_pipeline = TextureForensicsPipeline()

    def analyze_with_vllm(
        self,
        image: Image.Image,
        prompts: List[str],
        system_prompt: str
    ) -> Dict:
        """
        Run VLM-based analysis (Layer 3).

        Args:
            image: PIL Image
            prompts: List of analysis prompts
            system_prompt: System prompt for the model

        Returns:
            {
                "analysis": str,
                "score": float,
                "classification": str,
                "confidence": str
            }
        """
        client, model_name = get_client_and_name(self.model_key)

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        }

        all_responses = []
        overall_score = 0.0

        for prompt in prompts:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}, image_part],
                    },
                ],
                temperature=0.7,
                max_tokens=512,
            )

            analysis = response.choices[0].message.content
            all_responses.append(analysis)

            # Extract score
            score = self._extract_score(analysis)
            overall_score = max(overall_score, score)

        # Classification based on score (3-category)
        # 0-3: Real, 4-6.5: Uncertain/Manipulated, 6.5-10: AI Generated
        if overall_score <= 3.5:
            classification = "Real"
        elif overall_score >= 7.0:
            classification = "AI Generated"
        elif 5.0 <= overall_score < 7.0:
            # Mid-range suggests editing/manipulation rather than full synthesis
            classification = "AI Manipulated"
        else:
            classification = "Uncertain"

        # Determine confidence
        if overall_score > 8.5 or overall_score < 1.5:
            confidence = "high"
        elif overall_score > 7.0 or overall_score < 3.0:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "analysis": "\n".join(all_responses),
            "score": overall_score,
            "classification": classification,
            "confidence": confidence
        }

    def _extract_score(self, analysis: str) -> float:
        """Extract numeric score from analysis string."""
        import re

        match = re.search(
            r"(?:score|final score)\s*:?\s*\**\s*(\d+(?:\.\d+)?)",
            analysis,
            re.IGNORECASE,
        )
        if match:
            return float(match.group(1))

        match_slash = re.search(r"(\d+(?:\.\d+)?)/\d+", analysis)
        if match_slash:
            return float(match_slash.group(1))

        return 0.0

    def analyze_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompts: List[str] = None,
        system_prompt: str = None
    ) -> Dict:
        """
        Run full 3-layer analysis on an image.

        Args:
            image: Image as path, PIL Image, or numpy array
            prompts: Optional VLM prompts (uses default if None)
            system_prompt: Optional system prompt (uses default if None)

        Returns:
            {
                "layer1_physics": Dict or None,
                "layer2_texture": Dict or None,
                "layer3_vllm": Dict,
                "combined_verdict": str,
                "combined_score": float,
                "confidence": str,
                "explanation": str,
                "layer_agreement": Dict
            }
        """
        # Convert to PIL Image if needed
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Default prompts and system prompt
        if prompts is None:
            prompts = [
                """Assess the media and classify it into one of these categories:
- REAL (0-3): Authentic, unedited photograph
- MANIPULATED (4-6): Real photo with AI editing (face swap, inpainting, splicing, beauty filter artifacts)
- AI GENERATED (7-10): Fully synthetic/AI-created image

Explain your decision and rate your confidence as: High, Medium, or Low.
Key factors:
- Localized inconsistencies (splicing boundaries, lighting mismatches) → MANIPULATED
- Global synthetic patterns (impossible anatomy, diffusion smoothness, semantic errors) → AI GENERATED
- Natural sensor noise, authentic textures, coherent semantics → REAL"""
            ]

        if system_prompt is None:
            system_prompt = """You are a deepfake detection assistant. You will receive an image and must classify it.
Give a final 0-10 score:
0-3 = Real (authentic photograph)
4-6 = Manipulated (real photo with AI edits/splicing)
7-10 = AI Generated (fully synthetic)
Provide clear reasoning distinguishing localized editing from full synthesis."""

        # Initialize results
        layer1_result = None
        layer2_result = None

        # Run forensic layers if enabled
        if self.enable_all_layers:
            # Layer 1: Physics-Based Forensics
            try:
                layer1_result = self.physics_pipeline.analyze(pil_image)
            except Exception as e:
                layer1_result = {
                    "combined_verdict": "Error",
                    "confidence": "low",
                    "explanation": f"Physics layer failed: {str(e)}"
                }

            # Layer 2: Texture & Artifact Analysis
            try:
                layer2_result = self.texture_pipeline.analyze(pil_image)
            except Exception as e:
                layer2_result = {
                    "combined_verdict": "Error",
                    "confidence": "low",
                    "explanation": f"Texture layer failed: {str(e)}"
                }

        # Layer 3: VLM Analysis (with error handling)
        try:
            layer3_result = self.analyze_with_vllm(pil_image, prompts, system_prompt)
        except Exception as e:
            layer3_result = {
                "verdict": "Error",
                "confidence": "low",
                "analysis": f"VLM layer failed: {str(e)}",
                "score": 5.0,
                "classification": "Error"
            }

        # Combine results
        return self._combine_results(layer1_result, layer2_result, layer3_result)

    def _combine_results(
        self,
        layer1: Dict,
        layer2: Dict,
        layer3: Dict
    ) -> Dict:
        """
        Combine results from all layers using improved fusion logic.

        Key improvements:
        1. VLM weight is capped when uncorroborated by forensic layers
        2. "AI Manipulated" verdict is preserved (not flipped to "AI Generated")
        3. Probability-based scoring with better calibration

        Args:
            layer1: Physics-based results (or None)
            layer2: Texture-based results (or None)
            layer3: VLM results

        Returns:
            Combined analysis results
        """
        # Collect verdicts and confidences
        verdicts = {}
        confidences = {}

        if layer1 and layer1["combined_verdict"] not in ["Error", "Inconclusive", "Uncertain"]:
            verdicts["physics"] = layer1["combined_verdict"]
            confidences["physics"] = layer1["confidence"]

        if layer2 and layer2["combined_verdict"] not in ["Error", "Inconclusive", "Uncertain"]:
            verdicts["texture"] = layer2["combined_verdict"]
            confidences["texture"] = layer2["confidence"]

        if layer3["classification"] not in ["Error", "Uncertain"]:
            verdicts["vllm"] = layer3["classification"]
            confidences["vllm"] = layer3["confidence"]

        # --- IMPROVEMENT 1: Cap VLM weight when uncorroborated ---
        # Check if any forensic layer agrees with VLM
        forensic_verdicts = {k: v for k, v in verdicts.items() if k in ["physics", "texture"]}
        vllm_corroborated = False

        if "vllm" in verdicts:
            vllm_verdict = verdicts["vllm"]
            # Check for any agreement or semantic alignment
            for fv in forensic_verdicts.values():
                # Exact match or semantic alignment (both non-Real)
                if fv == vllm_verdict or (fv != "Real" and vllm_verdict != "Real"):
                    vllm_corroborated = True
                    break

        # Weighted voting with adjusted VLM weight
        confidence_weights = {"high": 3, "medium": 2, "low": 1}
        weighted_votes = []

        for layer, verdict in verdicts.items():
            if verdict not in ["Error", "Uncertain", "Inconclusive"]:
                weight = confidence_weights.get(confidences[layer], 1)

                # Cap VLM weight if uncorroborated
                if layer == "vllm" and not vllm_corroborated:
                    weight = min(weight, 1)  # Cap at 1 vote regardless of confidence

                weighted_votes.extend([verdict] * weight)

        # --- IMPROVEMENT 2: Preserve "AI Manipulated" verdict ---
        if not weighted_votes:
            combined_verdict = "Uncertain"
            combined_confidence = "low"
            consensus = False
            tie_detected = False
        else:
            # Count votes by category
            verdict_counts = Counter(weighted_votes)

            # Detect ties and apply deterministic tie-breaking
            tie_detected = False
            if len(verdict_counts) > 1:
                # Get the maximum vote count
                max_votes = verdict_counts.most_common(1)[0][1]
                # Find all verdicts with max votes (potential tie)
                tied_verdicts = [v for v, count in verdict_counts.items() if count == max_votes]

                if len(tied_verdicts) > 1:
                    tie_detected = True
                    # Conservative tie-breaking priority (most conservative first):
                    # 1. "AI Manipulated" - requires human review
                    # 2. "AI Generated" - definitive fake
                    # 3. "Real" - least conservative
                    verdict_priority = ["AI Manipulated", "AI Generated", "Real"]

                    # Select the highest priority verdict from tied ones
                    for priority_verdict in verdict_priority:
                        if priority_verdict in tied_verdicts:
                            combined_verdict = priority_verdict
                            break
                    else:
                        # Fallback (shouldn't happen with current verdicts)
                        combined_verdict = tied_verdicts[0]
                else:
                    # No tie, normal winner
                    combined_verdict = verdict_counts.most_common(1)[0][0]
            else:
                # Only one verdict type
                combined_verdict = verdict_counts.most_common(1)[0][0]

            # Special handling for "AI Manipulated" (can override tie-breaking)
            # If Layer 2 (texture) says "AI Manipulated" with medium+ confidence,
            # strongly prefer it (it's the only layer that can detect localized editing)
            if "texture" in verdicts and verdicts["texture"] == "AI Manipulated":
                texture_conf = confidences["texture"]
                if texture_conf in ["high", "medium"]:
                    # Give "AI Manipulated" strong preference
                    manipulated_votes = verdict_counts.get("AI Manipulated", 0)
                    ai_gen_votes = verdict_counts.get("AI Generated", 0)

                    # If Manipulated has any votes and isn't overwhelmingly outvoted
                    if manipulated_votes > 0 and manipulated_votes >= ai_gen_votes * 0.4:
                        combined_verdict = "AI Manipulated"

            # Check consensus
            unique_verdicts = set(v for v in verdicts.values() if v not in ["Error", "Uncertain", "Inconclusive"])
            consensus = len(unique_verdicts) == 1

            # Determine combined confidence
            if consensus and all(c == "high" for c in confidences.values()):
                combined_confidence = "high"
            elif consensus:
                combined_confidence = "medium"
            else:
                # Mixed results - calculate agreement ratio
                total_votes = sum(verdict_counts.values())
                majority_votes = verdict_counts.most_common(1)[0][1]
                agreement_ratio = majority_votes / total_votes

                if agreement_ratio > 0.7:
                    combined_confidence = "medium"
                else:
                    combined_confidence = "low"

            # Apply confidence penalty for ties (downgrade confidence level)
            # Ties indicate weak consensus and should lower our certainty
            if tie_detected:
                if combined_confidence == "high":
                    combined_confidence = "medium"
                elif combined_confidence == "medium":
                    combined_confidence = "low"
                # "low" stays "low"

        # --- IMPROVEMENT 3: Probability-based scoring ---
        # Convert verdicts to probability-like scores (0=Real, 1=AI)
        verdict_to_prob = {
            "Real": 0.1,
            "Uncertain": 0.5,
            "AI Manipulated": 0.7,
            "AI Generated": 0.9
        }

        # Calculate weighted probability
        total_weight = 0
        weighted_prob = 0

        for layer, verdict in verdicts.items():
            if verdict in verdict_to_prob:
                weight = confidence_weights.get(confidences[layer], 1)

                # Apply VLM cap
                if layer == "vllm" and not vllm_corroborated:
                    weight = min(weight, 1)

                weighted_prob += verdict_to_prob[verdict] * weight
                total_weight += weight

        final_prob = weighted_prob / total_weight if total_weight > 0 else 0.5

        # Convert probability to 0-10 score
        combined_score = final_prob * 10.0

        # Generate explanation
        explanations = []

        if layer1:
            explanations.append(f"Layer 1 (Physics): {layer1['explanation']} → {verdicts.get('physics', 'N/A')}")

        if layer2:
            explanations.append(f"Layer 2 (Texture): {layer2['explanation']} → {verdicts.get('texture', 'N/A')}")

        vllm_note = " (capped - uncorroborated)" if "vllm" in verdicts and not vllm_corroborated else ""
        explanations.append(f"Layer 3 (VLM): Score {layer3['score']}/10 → {layer3['classification']}{vllm_note}")

        # Note if AI Manipulated was preserved
        if combined_verdict == "AI Manipulated":
            explanations.append("⚠️ VERDICT: AI Manipulated preserved (Layer 2 detected splicing/editing)")

        # Add tie warning if detected
        if tie_detected:
            explanations.append("⚠️ TIE DETECTED: Multiple verdicts tied - conservative verdict selected, confidence downgraded")

        explanation = "\n\n".join(explanations)

        return {
            "layer1_physics": layer1,
            "layer2_texture": layer2,
            "layer3_vllm": layer3,
            "combined_verdict": combined_verdict,
            "combined_score": combined_score,
            "confidence": combined_confidence,
            "explanation": explanation,
            "layer_agreement": {
                "physics": verdicts.get("physics", "N/A"),
                "texture": verdicts.get("texture", "N/A"),
                "vllm": verdicts.get("vllm", "N/A"),
                "consensus": consensus,
                "vllm_corroborated": vllm_corroborated,
                "tie_detected": tie_detected
            }
        }

    def batch_analyze(
        self,
        images: List[Union[str, Image.Image]],
        prompts: List[str] = None,
        system_prompt: str = None,
        progress_callback = None
    ) -> List[Dict]:
        """
        Analyze multiple images in batch.

        Args:
            images: List of images (paths or PIL Images)
            prompts: Optional VLM prompts
            system_prompt: Optional system prompt
            progress_callback: Optional callback function(current, total)

        Returns:
            List of analysis results
        """
        results = []

        for i, image in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, len(images))

            result = self.analyze_image(image, prompts, system_prompt)
            result["image_index"] = i
            results.append(result)

        return results


# Backward compatibility: Keep original function signature
def analyze_single_image(
    image: Image.Image,
    prompts: List[str],
    system_prompt: str,
    model_key: str,
    enable_forensics: bool = True
) -> Dict:
    """
    Analyze a single image (backward compatible function).

    Args:
        image: PIL Image
        prompts: Analysis prompts
        system_prompt: System prompt
        model_key: Model key for VLM
        enable_forensics: If True, run all 3 layers. If False, VLM only.

    Returns:
        {
            "classification": str,
            "analysis": str,
            "score": float,
            "forensics": Dict (if enabled)
        }
    """
    detector = EnhancedDeepfakeDetector(
        model_key=model_key,
        enable_all_layers=enable_forensics
    )

    result = detector.analyze_image(image, prompts, system_prompt)

    # Return in old format for compatibility
    return {
        "classification": result["combined_verdict"],
        "analysis": result["explanation"],
        "score": result["combined_score"],
        "forensics": {
            "physics": result["layer1_physics"],
            "texture": result["layer2_texture"],
            "vllm": result["layer3_vllm"],
            "layer_agreement": result["layer_agreement"],
            "confidence": result["confidence"]
        }
    }
