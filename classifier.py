"""
Forensic classification with logit calibration.

This module implements glass-box classification using raw log-probabilities
extracted from VLM API responses, combined with forensic artifact analysis
(ELA + FFT) to detect AI-generated images.

Key Innovation:
    Instead of asking "Is this real or AI?" and parsing text responses (black-box),
    this classifier:
    1. Sends three images: Original + ELA + FFT
    2. Forces single-token output ("REAL" or "FAKE")
    3. Extracts raw logprobs from model internals
    4. Applies softmax normalization for calibrated confidence scores
    5. Uses data-driven threshold instead of arbitrary 50%

This approach eliminates "Timidity Bias" where models suspect AI generation
but don't cross the 50% threshold due to hedging behavior.
"""

import math
from typing import Dict, List, Union, Optional
from PIL import Image
import base64
import io
from openai import OpenAI


class ForensicClassifier:
    """
    Classify images using forensic artifacts and logit calibration.

    This classifier analyzes three forensic views of an image:
    1. Original photograph
    2. ELA (Error Level Analysis) map
    3. FFT (Fast Fourier Transform) spectrum

    And extracts raw log-probabilities to get true model confidence.
    """

    # Forensic system prompt
    SYSTEM_PROMPT = """You are a forensic signal processing unit. You analyze three input images:

1. Original Photograph
2. ELA (Error Level Analysis) Map
3. FFT (Fast Fourier Transform) Spectrum

Analysis Rules:
- If FFT shows a "Grid", "Starfield", or "Cross" pattern → FAKE
- If ELA shows uniform "Rainbow" static across the whole image → FAKE
- If Original shows physical inconsistencies (pupils, hands, reflections) → FAKE
- If FFT is a chaotic "Starburst" AND ELA is uniform dark/edge-noise → REAL

Question: Is this image REAL or FAKE?

A) Real (authentic photograph)
B) Fake (AI-generated or manipulated)

Answer with ONLY the letter (A or B). Nothing else."""

    # Token variations for A/B responses
    REAL_TOKENS = ['A', ' A', 'a', ' a']  # A = Real
    FAKE_TOKENS = ['B', ' B', 'b', ' b']  # B = Fake

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "dummy",
        threshold: float = 0.5
    ):
        """
        Initialize forensic classifier.

        Args:
            base_url: OpenAI-compatible API endpoint (e.g., "http://localhost:8000/v1/")
            model_name: Model identifier for API calls
            api_key: API key (default: "dummy" for vLLM)
            threshold: Classification threshold (default: 0.5)
                      Values > threshold → AI Generated
                      Values ≤ threshold → Authentic
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.threshold = threshold

    def _image_to_base64(self, image_bytes: bytes) -> str:
        """
        Convert image bytes to base64 data URI.

        Args:
            image_bytes: PNG or JPEG bytes

        Returns:
            Base64-encoded data URI string
        """
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    def _pil_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 data URI.

        Args:
            image: PIL Image object

        Returns:
            Base64-encoded data URI string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return self._image_to_base64(buffered.getvalue())

    def classify_image(
        self,
        original_bytes: bytes,
        ela_bytes: bytes,
        fft_bytes: bytes
    ) -> Dict:
        """
        Classify image using forensic artifacts and logit calibration.

        Args:
            original_bytes: Original image as PNG/JPEG bytes
            ela_bytes: ELA map as PNG bytes
            fft_bytes: FFT spectrum as PNG bytes

        Returns:
            {
                "is_ai": bool,
                "confidence_score": float (0.0-1.0),
                "raw_logits": {
                    "real": float,
                    "fake": float
                },
                "classification": str ("Authentic" or "AI-Generated"),
                "token_output": str (actual token from model),
                "threshold": float (threshold used for classification)
            }

        Raises:
            Exception: If API call fails or logprob parsing fails
        """
        # Convert images to base64
        original_uri = self._image_to_base64(original_bytes)
        ela_uri = self._image_to_base64(ela_bytes)
        fft_uri = self._image_to_base64(fft_bytes)

        # Construct message with all three images
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Original Photograph:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": original_uri}
                    },
                    {
                        "type": "text",
                        "text": "ELA (Error Level Analysis) Map:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": ela_uri}
                    },
                    {
                        "type": "text",
                        "text": "FFT (Fast Fourier Transform) Spectrum:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": fft_uri}
                    },
                    {
                        "type": "text",
                        "text": "Classify: REAL or FAKE?"
                    }
                ]
            }
        ]

        # API call with logprobs
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT}
                ] + messages,
                temperature=0.0,  # Deterministic for forensic analysis
                max_tokens=1,     # Force single token output
                logprobs=True,    # Enable logprob extraction
                top_logprobs=5    # Get top 5 tokens
            )

            # Parse logprobs
            result = self._parse_logprobs(response)
            result["threshold"] = self.threshold

            return result

        except Exception as e:
            # Graceful error handling with neutral classification
            return {
                "is_ai": False,
                "confidence_score": 0.5,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "classification": "Error",
                "token_output": None,
                "threshold": self.threshold,
                "error": str(e)
            }

    def classify_pil_image(
        self,
        original: Image.Image,
        ela: Image.Image,
        fft: Image.Image
    ) -> Dict:
        """
        Classify PIL Images directly (convenience method).

        Args:
            original: Original PIL Image
            ela: ELA map as PIL Image
            fft: FFT spectrum as PIL Image

        Returns:
            Classification result dict (same as classify_image)
        """
        # Convert PIL Images to bytes
        def pil_to_bytes(img: Image.Image) -> bytes:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return buffered.getvalue()

        original_bytes = pil_to_bytes(original)
        ela_bytes = pil_to_bytes(ela)
        fft_bytes = pil_to_bytes(fft)

        return self.classify_image(original_bytes, ela_bytes, fft_bytes)

    def _parse_logprobs(self, response) -> Dict:
        """
        Parse logprobs from API response.

        Handles tokenization variations (REAL vs Real vs real, with/without leading space).

        Algorithm:
            1. Extract top_logprobs from first token
            2. Scan for REAL/FAKE token variations
            3. Convert logprobs to linear probabilities: p = exp(logprob)
            4. Apply softmax normalization: p_fake / (p_fake + p_real)
            5. Apply threshold for binary classification

        Args:
            response: OpenAI API response object with logprobs

        Returns:
            Classification result dict

        Raises:
            Exception: If logprobs cannot be parsed
        """
        try:
            # Extract logprobs from first (and only) token
            logprobs_content = response.choices[0].logprobs.content[0]
            top_logprobs = logprobs_content.top_logprobs
            token_output = logprobs_content.token

            # Initialize scores in log space
            score_real = None
            score_fake = None

            # Collect all tokens for debugging
            all_tokens_debug = [(obj.token, obj.logprob) for obj in top_logprobs]

            # Scan top logprobs for REAL/FAKE tokens
            for logprob_obj in top_logprobs:
                token = logprob_obj.token
                logprob = logprob_obj.logprob

                # Check for REAL token variations
                if token in self.REAL_TOKENS and score_real is None:
                    score_real = logprob
                # Check for FAKE token variations
                elif token in self.FAKE_TOKENS and score_fake is None:
                    score_fake = logprob

            # If we didn't find both tokens, try to infer from actual output token
            if score_real is None or score_fake is None:
                # Use actual output token's logprob if it matches
                actual_logprob = logprobs_content.logprob

                if token_output in self.REAL_TOKENS:
                    score_real = actual_logprob
                elif token_output in self.FAKE_TOKENS:
                    score_fake = actual_logprob

            # Validate we found at least one token
            if score_real is None and score_fake is None:
                raise ValueError(
                    f"Could not find REAL or FAKE tokens in top_logprobs. "
                    f"Token output: '{token_output}', "
                    f"Top tokens: {all_tokens_debug}"
                )

            # If only one found, assign very low probability to the other
            if score_real is None:
                score_real = score_fake - 10.0  # exp(-10) ≈ 0.000045 relative probability
            if score_fake is None:
                score_fake = score_real - 10.0

            # Convert to linear space
            p_real = math.exp(score_real)
            p_fake = math.exp(score_fake)

            # Softmax normalization
            # This gives us calibrated probability: P(FAKE | image, ELA, FFT)
            confidence_fake = p_fake / (p_fake + p_real)

            # Apply threshold
            is_ai = confidence_fake > self.threshold
            classification = "AI-Generated" if is_ai else "Authentic"

            return {
                "is_ai": is_ai,
                "confidence_score": confidence_fake,
                "raw_logits": {
                    "real": score_real,
                    "fake": score_fake
                },
                "raw_probs": {
                    "real": p_real,
                    "fake": p_fake
                },
                "classification": classification,
                "token_output": token_output,
                "debug_tokens": all_tokens_debug  # For debugging token issues
            }

        except AttributeError as e:
            # Handle case where response doesn't have expected structure
            raise ValueError(
                f"Response missing expected logprobs structure: {e}"
            ) from e
        except Exception as e:
            # Re-raise other exceptions with context
            raise RuntimeError(
                f"Failed to parse logprobs: {e}"
            ) from e

    def set_threshold(self, threshold: float):
        """
        Update classification threshold.

        Args:
            threshold: New threshold value (0.0-1.0)
                      Values > threshold → AI Generated
                      Values ≤ threshold → Authentic

        Raises:
            ValueError: If threshold not in [0.0, 1.0]
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold must be between 0.0 and 1.0, got {threshold}"
            )
        self.threshold = threshold


def create_classifier_from_config(
    model_key: str,
    threshold: float = 0.5
) -> ForensicClassifier:
    """
    Create classifier from config.py MODEL_CONFIGS.

    Args:
        model_key: Key from MODEL_CONFIGS dict
        threshold: Classification threshold (default: 0.5)

    Returns:
        ForensicClassifier instance

    Example:
        >>> from config import MODEL_CONFIGS
        >>> classifier = create_classifier_from_config("Qwen3-VL-32B-Instruct")
        >>> result = classifier.classify_image(original, ela, fft)
    """
    from config import MODEL_CONFIGS

    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_key: {model_key}. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_key]

    return ForensicClassifier(
        base_url=config["base_url"],
        model_name=config["model_name"],
        api_key=config.get("api_key", "dummy"),
        threshold=threshold
    )


if __name__ == "__main__":
    # Example usage demonstration
    print("Forensic Classifier - Logit Calibration System")
    print("=" * 60)
    print()
    print("This module implements glass-box classification using:")
    print("  1. Forensic artifacts (ELA + FFT)")
    print("  2. Raw log-probability extraction")
    print("  3. Softmax normalization for calibrated confidence")
    print()
    print("Key Benefits:")
    print("  ✓ Eliminates 'Timidity Bias'")
    print("  ✓ True model confidence (not subjective text)")
    print("  ✓ Data-driven threshold optimization")
    print("  ✓ 2560x more efficient (1 token vs 512 tokens × 5 runs)")
    print()
    print("Usage Example:")
    print("-" * 60)
    print("""
    from detector import OSINTDetector
    from PIL import Image
    import io

    # Load image
    image = Image.open('photo.jpg')
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()

    # Create OSINT detector with SPAI
    detector = OSINTDetector(
        base_url="http://localhost:8000/v1",
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        detection_mode="spai_assisted"  # or "spai_standalone"
    )

    # Run detection
    result = detector.detect(img_bytes, debug=True)

    print(f"Classification: {result['tier']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"SPAI Score: {result['debug']['spai_score']:.3f}")
    print(f"Reasoning: {result['reasoning']}")
    """)
