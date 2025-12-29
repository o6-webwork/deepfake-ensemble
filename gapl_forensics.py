"""
GAPL (Generator-Aware Prototype Learning) Forensics for Deepfake Detection

This module integrates the GAPL framework for detecting AI-generated images.
GAPL learns forgery concepts as generator-aware prototypes that generalize
across different AI image generation models.

Repository: https://github.com/UltraCapture/GAPL
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Union
import io
import warnings
import sys
from pathlib import Path

# Add GAPL to Python path
GAPL_PATH = Path(__file__).parent / "gapl"
if GAPL_PATH.exists() and str(GAPL_PATH) not in sys.path:
    sys.path.insert(0, str(GAPL_PATH))


class GAPLDetector:
    """
    GAPL-based AI-generated image detector.

    Uses generator-aware prototype learning to detect synthetic images
    from various AI generation models.
    """

    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize GAPL detector.

        Args:
            model_path: Path to pretrained GAPL model weights (default: gapl/pretrained/checkpoint.pt)
            device: Device to run inference on ("cuda" or "cpu", auto-detect if None)
        """
        # Set default model path if not provided
        if model_path is None:
            model_path = str(GAPL_PATH / "pretrained" / "checkpoint.pt")

        self.model_path = model_path
        self.device = device  # Will be set in _load_model
        self.model = None
        self.is_available = False
        self.torch = None

        # Try to import GAPL dependencies
        try:
            import torch
            import torchvision
            self.torch = torch
            self.torchvision = torchvision

            # Auto-detect device if not specified
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self._load_model()
        except ImportError as e:
            warnings.warn(
                f"GAPL dependencies not available: {e}. "
                "Install with: pip install torch torchvision"
            )
            self.is_available = False

    def _load_model(self):
        """Load the pretrained GAPL model."""
        try:
            # Import GAPL model architecture
            try:
                import models as gapl_models
            except ImportError as e:
                warnings.warn(
                    f"GAPL models module not found: {e}. "
                    "Please ensure GAPL repository is cloned to gapl/ directory"
                )
                return

            # Check if model checkpoint exists
            if not Path(self.model_path).exists():
                warnings.warn(
                    f"GAPL checkpoint not found at {self.model_path}. "
                    "Please download from: https://huggingface.co/AbyssLumine/GAPL"
                )
                return

            # Determine device
            device_str = self.device if self.torch.cuda.is_available() and self.device == "cuda" else "cpu"
            device = self.torch.device(device_str)

            # Initialize model with correct device
            self.model = gapl_models.GAPLModel(
                fe_path=None,
                proto_path=None,
                freeze_backbone=False,
                device=device_str
            )

            # Load checkpoint
            print(f"[GAPL] Loading checkpoint from {self.model_path}...")
            checkpoint = self.torch.load(self.model_path, map_location='cpu')

            # Load model state and prototypes
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.model.load_prototype(checkpoint['prototype'])

            # Move to device and set to eval mode
            self.model.to(device)
            self.model.eval()

            self.device = device_str
            self.is_available = True
            print(f"[GAPL] Model loaded successfully on {device_str}")

        except Exception as e:
            import traceback
            warnings.warn(f"Failed to load GAPL model: {e}\n{traceback.format_exc()}")
            self.is_available = False

    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> 'torch.Tensor':
        """
        Preprocess image for GAPL inference.

        Args:
            image: PIL Image or numpy array

        Returns:
            Preprocessed tensor ready for model input
        """
        if not self.is_available:
            return None

        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # GAPL preprocessing (from inference_image.py)
        from torchvision import transforms

        # Use ImageNet normalization
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        tensor = transform(image).unsqueeze(0)

        # Move to correct device
        device = self.torch.device(self.device)
        tensor = tensor.to(device)

        return tensor

    def detect(self, image: Union[Image.Image, np.ndarray]) -> Dict:
        """
        Run GAPL detection on an image.

        Args:
            image: PIL Image or numpy array

        Returns:
            {
                "is_ai_generated": bool,
                "confidence": float (0-1),
                "generator_prototypes": list of detected generator types (if available),
                "raw_score": float,
                "explanation": str
            }
        """
        if not self.is_available or self.model is None:
            return {
                "is_ai_generated": None,
                "confidence": 0.0,
                "generator_prototypes": [],
                "raw_score": 0.0,
                "explanation": "GAPL model not available. Please install dependencies and download model weights."
            }

        try:
            # Preprocess image
            tensor = self.preprocess_image(image)

            # Run inference
            with self.torch.no_grad():
                # Run GAPL model (returns raw logit score)
                output = self.model(tensor)

                # Apply sigmoid to get probability (0-1 scale)
                # Higher score = more likely AI-generated
                score = output.sigmoid().item()

            # Threshold for classification (matching inference_image.py)
            threshold = 0.5
            is_ai_generated = score > threshold

            # Calculate confidence based on distance from threshold
            confidence = abs(score - threshold) * 2  # Scale to 0-1

            # Build explanation
            label_str = "AI-Generated" if is_ai_generated else "Real"
            explanation = f"GAPL prediction: {label_str} (score: {score:.3f}, confidence: {confidence:.2%})"

            return {
                "is_ai_generated": is_ai_generated,
                "confidence": confidence,
                "generator_prototypes": [],  # GAPL doesn't directly output generator types
                "raw_score": score,
                "explanation": explanation
            }

        except Exception as e:
            import traceback
            return {
                "is_ai_generated": None,
                "confidence": 0.0,
                "generator_prototypes": [],
                "raw_score": 0.0,
                "explanation": f"GAPL detection failed: {str(e)}\n{traceback.format_exc()}"
            }


class GAPLForensicsPipeline:
    """
    GAPL forensics pipeline for integration with NexInspect detector.

    Provides a unified interface matching other forensic layers
    (physics_forensics.py, texture_forensics.py).
    """

    def __init__(self, model_path: str = None, device: str = "cuda"):
        """
        Initialize GAPL forensics pipeline.

        Args:
            model_path: Path to pretrained GAPL model weights
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.detector = GAPLDetector(model_path=model_path, device=device)

    def _calibrate_confidence(self, raw_conf: float, temperature: float = 1.5) -> float:
        """
        Apply temperature scaling to GAPL confidence to reduce overconfidence.

        GAPL uses sigmoid(logits) which can be overconfident on out-of-distribution data
        (HADR scenes, military images, low-light conditions).

        Args:
            raw_conf: Raw GAPL confidence (0-1)
            temperature: T > 1 reduces confidence (default 1.5 for OOD robustness)

        Returns:
            Calibrated confidence (0-1)
        """
        import math

        # Avoid division by zero and extreme values
        if raw_conf >= 0.9999:
            logit = 9.0
        elif raw_conf <= 0.0001:
            logit = -9.0
        else:
            logit = math.log(raw_conf / (1.0 - raw_conf))

        # Apply temperature scaling
        scaled_logit = logit / temperature

        # Convert back to probability
        calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))

        return calibrated

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Run GAPL-based forensic analysis.

        Args:
            image_input: Path to image, PIL Image, or numpy array

        Returns:
            {
                "gapl_detection": Dict,
                "combined_verdict": str ("Real", "AI Generated", or "Uncertain"),
                "confidence": str ("high", "medium", "low"),
                "explanation": str
            }
        """
        # Load image if path provided
        if isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            if image_input.shape[2] == 3:  # Assume BGR from OpenCV
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_rgb)
            else:
                image = Image.fromarray(image_input)
        else:
            image = image_input

        # Run GAPL detection
        gapl_result = self.detector.detect(image)

        # Convert to unified verdict format
        if gapl_result["is_ai_generated"] is None:
            # Model not available
            combined_verdict = "Uncertain"
            confidence_level = "low"
            explanation = "GAPL analysis unavailable: " + gapl_result["explanation"]
        else:
            # Map GAPL confidence to verdict
            raw_confidence = gapl_result["confidence"]

            # Apply temperature scaling to reduce overconfidence on OOD data
            calibrated_confidence = self._calibrate_confidence(raw_confidence, temperature=1.5)

            if gapl_result["is_ai_generated"]:
                combined_verdict = "AI Generated"
                if calibrated_confidence >= 0.7:
                    confidence_level = "high"
                elif calibrated_confidence >= 0.4:
                    confidence_level = "medium"
                else:
                    confidence_level = "low"
            else:
                combined_verdict = "Real"
                if calibrated_confidence >= 0.7:
                    confidence_level = "high"
                elif calibrated_confidence >= 0.4:
                    confidence_level = "medium"
                else:
                    confidence_level = "low"

            # Build explanation
            explanation = gapl_result["explanation"]
            if gapl_result["generator_prototypes"]:
                explanation += f"\n  Potential generators: {', '.join(gapl_result['generator_prototypes'])}"
            explanation += f"\n  Raw GAPL score: {gapl_result['raw_score']:.3f}"
            explanation += f"\n  Calibrated confidence (T=1.5): {calibrated_confidence:.3f}"

        return {
            "gapl_detection": gapl_result,
            "combined_verdict": combined_verdict,
            "confidence": confidence_level,
            "explanation": explanation
        }


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gapl_forensics.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Initialize pipeline
    pipeline = GAPLForensicsPipeline()

    # Analyze image
    result = pipeline.analyze(image_path)

    # Print results
    print("\n=== GAPL Forensic Analysis ===")
    print(f"Verdict: {result['combined_verdict']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Explanation: {result['explanation']}")

    if result['gapl_detection']['generator_prototypes']:
        print(f"Detected Generators: {', '.join(result['gapl_detection']['generator_prototypes'])}")
