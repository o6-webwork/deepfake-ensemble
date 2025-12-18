"""
SPAI (Spectral AI-Generated Image Detector) wrapper module.

This module provides a clean interface to the SPAI CVPR2025 model for deepfake detection
using spectral learning. Replaces ELA/FFT forensic analysis with state-of-the-art
frequency-domain detection.

SPAI analyzes the spectral distribution of images using masked feature modeling with
Vision Transformers to detect AI-generated content with high accuracy.
"""

import torch
import cv2
import numpy as np
from PIL import Image
import io
import logging
from pathlib import Path
from typing import Dict, Optional

# Import SPAI components from spai/ directory
import sys
sys.path.insert(0, str(Path(__file__).parent / 'spai'))

try:
    from spai.config import get_config
    from spai.models import build_cls_model
    from spai.utils import load_pretrained
    from spai.data.data_finetune import build_transform
except ImportError as e:
    raise ImportError(
        f"Failed to import SPAI modules. Ensure spai/ directory exists with required files. Error: {e}"
    )

# Configure logging
logger = logging.getLogger(__name__)


class SPAIDetector:
    """
    SPAI spectral deepfake detector.

    Uses masked feature modeling with Vision Transformers to analyze frequency-domain
    patterns characteristic of AI-generated images.

    Replaces traditional ELA/FFT forensics with learned spectral representations.
    """

    def __init__(
        self,
        config_path: str = "spai/configs/spai.yaml",
        weights_path: str = "spai/weights/spai.pth",
        device: Optional[str] = None
    ):
        """
        Initialize SPAI model.

        Args:
            config_path: Path to SPAI YAML config file
            weights_path: Path to pre-trained SPAI weights
            device: Device to run model on ('cuda' or 'cpu'). Auto-detects if None.

        Raises:
            FileNotFoundError: If weights file not found
            RuntimeError: If model initialization fails
        """
        # Validate paths
        config_path = Path(config_path)
        weights_path = Path(weights_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"SPAI config not found at {config_path}. "
                f"Ensure spai/configs/spai.yaml exists."
            )

        if not weights_path.exists():
            raise FileNotFoundError(
                f"SPAI weights not found at {weights_path}.\n"
                f"Download from: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view\n"
                f"Place in: spai/weights/spai.pth"
            )

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Show device info prominently
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 SPAI using GPU: {gpu_name}")
            print(f"🚀 SPAI using GPU: {gpu_name}")  # Also print for visibility
        else:
            logger.warning(f"⚠️ SPAI using CPU (SLOW) - CUDA not available")
            print(f"⚠️ SPAI using CPU (SLOW) - CUDA not available")

        # Load config
        try:
            self.config = get_config({
                "cfg": str(config_path),
                "batch_size": 1,
                "opts": []
            })
        except Exception as e:
            raise RuntimeError(f"Failed to load SPAI config: {e}")

        # Build model
        try:
            self.model = build_cls_model(self.config)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to build SPAI model: {e}")

        # Load pre-trained weights
        try:
            load_pretrained(
                self.config,
                self.model,
                logger,
                checkpoint_path=str(weights_path),
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SPAI weights: {e}")

        self.model.eval()

        # Build transform pipeline
        try:
            self.transform = build_transform(is_train=False, config=self.config)
        except Exception as e:
            raise RuntimeError(f"Failed to build SPAI transform: {e}")

        logger.info("SPAI initialized successfully")

    def analyze(
        self,
        image_bytes: bytes,
        generate_heatmap: bool = True,
        alpha: float = 0.6,
        max_size: Optional[int] = 1280
    ) -> Dict:
        """
        Run SPAI spectral analysis on image.

        Args:
            image_bytes: Input image as bytes (PNG/JPEG)
            generate_heatmap: If True, generate blended attention overlay
            alpha: Transparency for blending (0.0-1.0)
                   Higher alpha = more original image visible
                   Default 0.6 = 60% original, 40% heatmap
            max_size: Maximum resolution for longest edge (512-2048 or None for original)
                     Higher = more accurate but slower. Default 1280.

        Returns:
            {
                "spai_score": float,         # 0.0-1.0 (AI generation probability)
                "spai_prediction": str,      # "Real" or "AI Generated"
                "tier": str,                 # "Authentic" / "Suspicious" / "Deepfake"
                "heatmap_bytes": bytes,      # Blended overlay PNG (if requested)
                "analysis_text": str         # Human-readable analysis with confidence levels
            }

        Raises:
            ValueError: If image cannot be loaded
            RuntimeError: If inference fails
        """
        import time
        total_start = time.time()

        # Clone config and set resolution
        t0 = time.time()
        config = self.config.clone()
        config.defrost()
        config.TEST.MAX_SIZE = max_size  # None = original resolution
        if generate_heatmap:
            config.MODEL.RESOLUTION_MODE = "arbitrary"  # Required for attention extraction
        config.freeze()
        print(f"⏱️ Config setup: {time.time() - t0:.3f}s")

        # Rebuild transform with updated config
        t0 = time.time()
        transform = build_transform(is_train=False, config=config)
        print(f"⏱️ Build transform: {time.time() - t0:.3f}s")

        # Clear CUDA cache if using GPU (improves performance)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Load image
        t0 = time.time()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
        print(f"⏱️ Load image: {time.time() - t0:.3f}s")

        # Preprocess - albumentations requires named 'image=' argument
        t0 = time.time()
        try:
            img_np = np.array(pil_image)
            tensor = transform(image=img_np)["image"].unsqueeze(0).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess image: {e}")
        print(f"⏱️ Preprocess: {time.time() - t0:.3f}s")

        # Run inference - branch based on heatmap requirement (like spai/app.py)
        t0 = time.time()
        heatmap_bytes = None
        attention_masks = None

        try:
            with torch.no_grad():
                if generate_heatmap and config.MODEL.RESOLUTION_MODE == "arbitrary":
                    # ONE model call with heatmap generation (matches spai/app.py line 148-152)
                    import tempfile
                    with tempfile.TemporaryDirectory() as temp_export_dir:
                        export_path = Path(temp_export_dir)

                        output, attention_masks = self.model(
                            x=[tensor],  # Wrapped in list with 'x=' parameter
                            feature_extraction_batch_size=config.MODEL.FEATURE_EXTRACTION_BATCH,
                            export_dirs=[export_path]
                        )
                        score = torch.sigmoid(output).item()
                        print(f"⏱️ Inference WITH heatmap (single pass): {time.time() - t0:.3f}s")

                        # Generate blended heatmap immediately while tempdir exists
                        if attention_masks and len(attention_masks) > 0:
                            t_blend = time.time()
                            heatmap_bytes = self._blend_heatmap_from_masks(
                                attention_masks[0],
                                pil_image,
                                alpha
                            )
                            print(f"  ⏱️ Blend heatmap: {time.time() - t_blend:.3f}s")
                else:
                    # Simple inference without heatmap (matches spai/app.py line 158)
                    output = self.model(tensor)
                    score = torch.sigmoid(output).item()
                    print(f"⏱️ Inference WITHOUT heatmap: {time.time() - t0:.3f}s")

        except Exception as e:
            raise RuntimeError(f"SPAI inference failed: {e}")

        # Determine prediction
        prediction = "AI Generated" if score >= 0.5 else "Real"

        # Map to three-tier system
        tier = self._map_score_to_tier(score)

        # Format analysis text
        analysis_text = self._format_analysis(score, prediction, tier)

        total_time = time.time() - total_start
        print(f"⏱️ TOTAL SPAI analyze(): {total_time:.3f}s")
        print(f"{'='*60}")

        return {
            "spai_score": score,
            "spai_prediction": prediction,
            "tier": tier,
            "heatmap_bytes": heatmap_bytes,
            "analysis_text": analysis_text,
            "timing": {  # Add timing data to result for UI display
                "total": total_time,
                "device": str(self.device)
            }
        }

    def _blend_heatmap_from_masks(
        self,
        mask_obj,
        original_image: Image.Image,
        alpha: float
    ) -> bytes:
        """
        Blend attention mask overlay with original image.

        This extracts the heatmap from the mask object (already generated by model)
        and blends it with the original image.

        Args:
            mask_obj: Attention mask object from model (has .overlay attribute)
            original_image: Original PIL Image
            alpha: Transparency weight (0.0-1.0, 0.6 = 60% original, 40% heatmap)

        Returns:
            PNG bytes of blended overlay

        Raises:
            RuntimeError: If blending fails
        """
        try:
            # Check if the overlay file was created by the model
            if not hasattr(mask_obj, 'overlay') or not mask_obj.overlay or not mask_obj.overlay.exists():
                raise RuntimeError("Model did not generate overlay file")

            # Use create_transparent_overlay pattern from spai/app.py
            # Convert Original PIL to OpenCV format (RGB)
            background = np.array(original_image)

            # Load the Overlay/Heatmap from disk (generated by model)
            foreground = cv2.imread(str(mask_obj.overlay))

            if foreground is None:
                raise RuntimeError(f"Failed to read overlay file at {mask_obj.overlay}")

            # Convert BGR (OpenCV default) to RGB to match PIL
            foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)

            # Resize Foreground to match Background exactly
            if foreground.shape[:2] != background.shape[:2]:
                foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

            # Blend: alpha = weight of original, beta = weight of heatmap
            beta = 1.0 - alpha
            blended = cv2.addWeighted(background, alpha, foreground, beta, 0)

            # Convert RGB to BGR for encoding
            blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

            # Encode as PNG
            success, png_bytes = cv2.imencode('.png', blended_bgr)

            if not success:
                raise RuntimeError("Failed to encode blended image as PNG")

            return png_bytes.tobytes()

        except Exception as e:
            raise RuntimeError(f"Heatmap blending failed: {e}")


    def _map_score_to_tier(self, score: float) -> str:
        """
        Map SPAI score to three-tier risk classification.

        Args:
            score: SPAI score (0.0-1.0, higher = more likely AI-generated)

        Returns:
            "Authentic", "Suspicious", or "Deepfake"
        """
        if score >= 0.9:
            return "Deepfake"
        elif score >= 0.5:
            return "Suspicious"
        else:
            return "Authentic"

    def _format_analysis(
        self,
        score: float,
        prediction: str,
        tier: str
    ) -> str:
        """
        Format SPAI analysis as human-readable text report.

        Args:
            score: SPAI score (0.0-1.0)
            prediction: "Real" or "AI Generated"
            tier: Risk tier classification

        Returns:
            Formatted analysis text
        """
        analysis = f"""--- SPAI SPECTRAL ANALYSIS ---

Prediction: {prediction}
SPAI Score: {score:.3f} (0.0=Real, 1.0=AI-Generated)

Spectral Analysis Summary:
The image's frequency spectrum was analyzed using masked feature modeling with
Vision Transformer architecture (CVPR2025). The spectral reconstruction similarity
score indicates {self._get_likelihood_text(score)}.

"""

        # Add interpretation based on score
        if score >= 0.9:
            analysis += "🚨 HIGH CONFIDENCE AI-GENERATED: Strong spectral artifacts detected.\n"
            analysis += "   The frequency distribution shows clear signatures of generative models.\n"
        elif score >= 0.7:
            analysis += "⚠️ LIKELY AI-GENERATED: Moderate spectral inconsistencies found.\n"
            analysis += "   Frequency patterns deviate from natural image distributions.\n"
        elif score >= 0.5:
            analysis += "⚠️ SUSPICIOUS: Subtle spectral anomalies present.\n"
            analysis += "   Some frequency components suggest possible AI generation.\n"
        elif score >= 0.3:
            analysis += "✓ LIKELY REAL: Spectral patterns consistent with natural images.\n"
            analysis += "   Frequency distribution matches authentic photography characteristics.\n"
        else:
            analysis += "✅ HIGH CONFIDENCE REAL: Natural spectral distribution detected.\n"
            analysis += "   Frequency patterns strongly indicate authentic capture.\n"

        analysis += "\nNote: SPAI analyzes frequency-domain patterns invisible to human perception.\n"
        analysis += "      These spectral signatures are characteristic of AI generation processes."

        return analysis

    def _get_likelihood_text(self, score: float) -> str:
        """
        Convert SPAI score to likelihood description.

        Args:
            score: SPAI score (0.0-1.0)

        Returns:
            Human-readable likelihood text
        """
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

    def __repr__(self) -> str:
        """String representation of SPAIDetector."""
        return f"SPAIDetector(device={self.device}, model_loaded={self.model is not None})"
