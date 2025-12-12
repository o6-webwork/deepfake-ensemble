"""
Forensic artifact generation for deepfake detection.

This module generates Error Level Analysis (ELA) and Fast Fourier Transform (FFT)
forensic maps to expose AI generation signatures in images.

Scientific Rationale:
    - ELA: Detects compression inconsistencies. AI-generated images have uniform
      compression levels (rainbow static), while real photos show varying compression
      (dark regions with edge noise) due to camera sensor behavior.

    - FFT: Reveals frequency domain patterns. GANs and diffusion models introduce
      periodic artifacts (grid/starfield/cross patterns), while real photos exhibit
      chaotic starburst patterns in frequency space.
"""

import cv2
import numpy as np
from PIL import Image
import io
import exifread
from typing import Union, Tuple, Dict, Optional


class ArtifactGenerator:
    """
    Generate forensic artifacts for image authentication analysis.

    This class provides static methods to create ELA and FFT maps that reveal
    forensic signatures indicative of AI generation or manipulation.
    """

    @staticmethod
    def generate_ela(
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        quality: int = 90,
        scale_factor: int = 15
    ) -> bytes:
        """
        Generate Error Level Analysis (ELA) map.

        ELA highlights compression inconsistencies by comparing the original
        image to a recompressed version. Uniform compression (AI-generated)
        appears as rainbow static; varying compression (real photos) shows
        dark regions with edge noise.

        Args:
            image_input: Input image as file path, PIL Image, or numpy array
            quality: JPEG compression quality for recompression (default: 90)
                    Lower quality = more visible differences
                    Range: 0-100
            scale_factor: Amplification factor for visibility (default: 15)
                         Higher value = more visible differences
                         Typical range: 10-20

        Returns:
            PNG-encoded bytes of the ELA map (can be displayed or saved)

        Algorithm:
            1. Load original image as numpy array
            2. Compress to JPEG at specified quality in memory
            3. Decode the compressed JPEG back to array
            4. Compute absolute difference: |Original - Compressed|
            5. Amplify differences: diff * scale_factor
            6. Clip to valid range [0, 255]
            7. Return as PNG bytes

        Example:
            >>> ela_bytes = ArtifactGenerator.generate_ela('photo.jpg')
            >>> with open('photo_ela.png', 'wb') as f:
            ...     f.write(ela_bytes)
        """
        # Load image to numpy array
        if isinstance(image_input, str):
            original = cv2.imread(image_input)
            if original is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        elif isinstance(image_input, bytes):
            # Decode bytes to numpy array
            nparr = np.frombuffer(image_input, np.uint8)
            original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if original is None:
                raise ValueError("Could not decode image from bytes")
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to OpenCV format (RGB -> BGR)
            original = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            original = image_input.copy()
        else:
            raise TypeError(
                "image_input must be str (path), bytes, PIL.Image, or numpy.ndarray"
            )

        # Validate quality parameter
        if not 0 <= quality <= 100:
            raise ValueError(f"quality must be between 0 and 100, got {quality}")

        # Compress to JPEG in memory
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, compressed_bytes = cv2.imencode('.jpg', original, encode_params)

        if not success:
            raise RuntimeError("Failed to encode image as JPEG")

        # Decode compressed image back to array
        compressed = cv2.imdecode(compressed_bytes, cv2.IMREAD_COLOR)

        if compressed is None:
            raise RuntimeError("Failed to decode compressed JPEG")

        # Ensure dimensions match (should always match, but safety check)
        if original.shape != compressed.shape:
            raise RuntimeError(
                f"Shape mismatch: original {original.shape} vs "
                f"compressed {compressed.shape}"
            )

        # Compute absolute difference
        diff = cv2.absdiff(original, compressed).astype('float')

        # Scale for visibility
        ela = np.clip(diff * scale_factor, 0, 255).astype('uint8')

        # Encode as PNG (lossless for artifact preservation)
        success, png_bytes = cv2.imencode('.png', ela)

        if not success:
            raise RuntimeError("Failed to encode ELA map as PNG")

        return png_bytes.tobytes()

    @staticmethod
    def generate_fft(
        image_input: Union[str, bytes, Image.Image, np.ndarray]
    ) -> bytes:
        """
        Generate Fast Fourier Transform (FFT) magnitude spectrum.

        FFT reveals frequency domain patterns. AI-generated images often
        exhibit grid, starfield, or cross patterns due to GAN/Diffusion
        architecture artifacts. Real photos show chaotic starburst patterns
        characteristic of natural scenes.

        Args:
            image_input: Input image as file path, PIL Image, or numpy array

        Returns:
            PNG-encoded bytes of the FFT magnitude spectrum

        Algorithm:
            1. Convert image to grayscale
            2. Convert to float32 for DFT computation
            3. Compute 2D Discrete Fourier Transform (DFT)
            4. Shift zero-frequency component to center (fftshift)
            5. Compute magnitude spectrum from complex output
            6. Apply log transform: 20 * log(magnitude + 1)
               - Log scale makes patterns more visible
               - +1 prevents log(0) errors
            7. Normalize to 0-255 range for visualization
            8. Return as PNG bytes

        Frequency Domain Signatures:
            - Grid patterns: Regular GAN artifacts
            - Starfield/Cross: Diffusion model artifacts
            - Chaotic starburst: Natural image frequencies

        Example:
            >>> fft_bytes = ArtifactGenerator.generate_fft('photo.jpg')
            >>> with open('photo_fft.png', 'wb') as f:
            ...     f.write(fft_bytes)
        """
        # Load and convert to grayscale
        if isinstance(image_input, str):
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        elif isinstance(image_input, bytes):
            # Decode bytes to numpy array, then convert to grayscale
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from bytes")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to grayscale
            if image_input.mode == 'L':
                gray = np.array(image_input)
            else:
                gray = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2GRAY)
        elif isinstance(image_input, np.ndarray):
            # Convert to grayscale if needed
            if len(image_input.shape) == 2:
                gray = image_input.copy()
            elif len(image_input.shape) == 3:
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(
                    f"Invalid image array shape: {image_input.shape}"
                )
        else:
            raise TypeError(
                "image_input must be str (path), bytes, PIL.Image, or numpy.ndarray"
            )

        # Convert to float32 for DFT
        gray_float = np.float32(gray)

        # Compute DFT
        # Output is complex array with 2 channels: real and imaginary
        dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Shift zero-frequency component to center
        # This places DC component (average brightness) at image center
        dft_shift = np.fft.fftshift(dft)

        # Compute magnitude spectrum from complex DFT
        # magnitude = sqrt(real^2 + imaginary^2)
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        # Apply log transform for better visibility
        # Log scale compresses dynamic range, making subtle patterns visible
        magnitude_log = 20 * np.log(magnitude + 1)  # +1 to avoid log(0)

        # Normalize to 0-255 range for visualization
        magnitude_normalized = cv2.normalize(
            magnitude_log,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )
        fft_image = np.uint8(magnitude_normalized)

        # Encode as PNG
        success, png_bytes = cv2.imencode('.png', fft_image)

        if not success:
            raise RuntimeError("Failed to encode FFT spectrum as PNG")

        return png_bytes.tobytes()

    @staticmethod
    def extract_metadata(
        image_input: Union[str, bytes, Image.Image]
    ) -> Dict[str, str]:
        """
        Extract EXIF metadata from image.

        Args:
            image_input: Input image as file path, bytes, or PIL Image

        Returns:
            Dictionary mapping EXIF tag names to values
            Empty dict if no EXIF data found

        Example:
            >>> metadata = ArtifactGenerator.extract_metadata('photo.jpg')
            >>> print(metadata.get('Image Make'))  # Camera manufacturer
            'Canon'
        """
        try:
            # Convert to file-like object for exifread
            if isinstance(image_input, str):
                with open(image_input, 'rb') as f:
                    tags = exifread.process_file(f, details=False)
            elif isinstance(image_input, bytes):
                tags = exifread.process_file(io.BytesIO(image_input), details=False)
            elif isinstance(image_input, Image.Image):
                # Convert PIL Image to bytes
                img_bytes = io.BytesIO()
                # Save with original format if available, otherwise use PNG
                img_format = image_input.format if image_input.format else 'PNG'
                image_input.save(img_bytes, format=img_format)
                img_bytes.seek(0)
                tags = exifread.process_file(img_bytes, details=False)
            else:
                raise TypeError(
                    "image_input must be str (path), bytes, or PIL.Image"
                )

            # Convert IfdTag objects to strings
            metadata = {
                str(tag): str(value)
                for tag, value in tags.items()
                if not tag.startswith('Thumbnail')  # Skip thumbnail tags
            }

            return metadata

        except Exception as e:
            # Return empty dict on error (image might not have EXIF)
            return {}

    @staticmethod
    def compute_ela_variance(ela_bytes: bytes) -> float:
        """
        Compute variance score from ELA map.

        CRITICAL UPDATE (Dec 12, 2025 - Forensic Expert Review):
        Low variance is INCONCLUSIVE on social media imagery due to aggressive
        platform re-compression (WhatsApp, Facebook, Twitter).

        ELA Variance Interpretation:
        - Low variance (<2.0): INCONCLUSIVE on social media
          * Platforms like WhatsApp/Facebook crush quantization tables
          * Real photos from WhatsApp will have variance ~0.5
          * DO NOT auto-flag as AI based on this alone
        - High variance (≥2.0): Potential manipulation/splicing
          * Indicates inconsistent compression across regions
          * Look for LOCAL inconsistencies (bright patch on dark background)

        VLM Guidance: Focus on SPATIAL PATTERNS (local inconsistencies), not
        global uniformity. A uniformly low ELA does not prove AI generation.

        Args:
            ela_bytes: PNG-encoded ELA map bytes from generate_ela()

        Returns:
            Standard deviation of pixel values across all channels

        Example:
            >>> ela_bytes = ArtifactGenerator.generate_ela('photo.jpg')
            >>> variance = ArtifactGenerator.compute_ela_variance(ela_bytes)
            >>> if variance < 2.0:
            ...     print("Low variance - INCONCLUSIVE on social media")
            >>> # VLM should look for LOCAL patterns instead
        """
        # Decode ELA image from PNG bytes
        nparr = np.frombuffer(ela_bytes, np.uint8)
        ela_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if ela_img is None:
            raise ValueError("Failed to decode ELA image bytes")

        # Compute standard deviation across all channels
        # Convert to float for accurate std calculation
        variance = np.std(ela_img.astype('float'))

        return float(variance)

    @staticmethod
    def generate_fft_preprocessed(
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        target_size: int = 512,
        apply_highpass: bool = True
    ) -> Tuple[bytes, Dict[str, any]]:
        """
        Generate FFT with preprocessing for improved pattern detection.

        CRITICAL FIXES (Dec 12, 2025 - Forensic Expert Review):
        1. NEVER resize - use center crop + padding to preserve pixel-level artifacts
        2. Mask DC component + central axes to remove social media "JPEG Cross"
        3. 5-Sigma peak detection to distinguish AI grid stars from natural grain

        Preprocessing steps:
        1. Center crop (if ≥512×512) or pad (if <512×512) - NO RESIZING
        2. Apply high-pass filter to remove DC bias from natural lighting
        3. Compute FFT magnitude spectrum
        4. Mask DC component (5px radius) + central axes (±1px) BEFORE normalization
        5. Normalize to 0-255 range
        6. Apply 5-Sigma adaptive thresholding for entropy-aware peak detection

        IMPORTANT: Linear interpolation acts as a low-pass filter, destroying
        high-frequency AI artifacts (checkerboard patterns from Transpose Convolutions).
        We use center crop + reflection padding instead.

        Social Media Fix: Twitter/Telegram images with rectangular borders create
        massive white crosses in FFT (100× stronger than AI artifacts). We mask
        the central axes to prevent VLM fixation on this artifact.

        5-Sigma Peak Detection:
        - AI Grid Stars: BRIGHT peaks that survive 5σ threshold (sparse, high-energy)
        - Real Grain: NOISY but DIMMER pixels filtered out by 5σ (diffuse, low-energy)
        - High Peak Count (>2000): Overwhelmingly textured real image with natural grain
        - Low Peak Count (<20): Clean natural image without artifacts
        - Medium Peak Count (20-2000): Suspected AI with high-frequency grid stars

        Args:
            image_input: Input image as file path, bytes, PIL Image, or numpy array
            target_size: Target dimension for crop/pad (default: 512)
            apply_highpass: Whether to apply high-pass filter (default: True)

        Returns:
            Tuple of (fft_bytes, metrics_dict):
                - fft_bytes: PNG-encoded FFT spectrum
                - metrics_dict: {
                    'pattern_type': str,  # 'Natural/Chaotic (High Entropy)',
                                          # 'High Freq Artifacts (Suspected AI)',
                                          # 'Natural/Clean'
                    'peaks_detected': int,
                    'peak_threshold': float  # Mean + 5*StdDev
                  }

        Example:
            >>> fft_bytes, metrics = ArtifactGenerator.generate_fft_preprocessed('photo.jpg')
            >>> print(f"Pattern: {metrics['pattern_type']}")
            >>> # Output: "High Freq Artifacts (Suspected AI)" or "Natural/Chaotic (High Entropy)"
        """
        # Load and convert to grayscale (same as original generate_fft)
        if isinstance(image_input, str):
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        elif isinstance(image_input, bytes):
            # Decode bytes to numpy array, then convert to grayscale
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from bytes")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif isinstance(image_input, Image.Image):
            if image_input.mode == 'L':
                gray = np.array(image_input)
            else:
                gray = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2GRAY)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                gray = image_input.copy()
            elif len(image_input.shape) == 3:
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"Invalid image array shape: {image_input.shape}")
        else:
            raise TypeError(
                "image_input must be str (path), bytes, PIL.Image, or numpy.ndarray"
            )

        # Step 1: Center crop + padding (CRITICAL FIX - Dec 12, 2025)
        # NEVER resize - interpolation acts as low-pass filter and destroys
        # pixel-level AI artifacts (checkerboard patterns from Transpose Convolutions)
        h, w = gray.shape
        crop_size = target_size

        # Calculate crop coordinates with bounds checking
        start_y = max(0, (h - crop_size) // 2)
        start_x = max(0, (w - crop_size) // 2)
        end_y = min(h, start_y + crop_size)
        end_x = min(w, start_x + crop_size)

        # Extract crop
        gray_crop = gray[start_y:end_y, start_x:end_x]

        # Reflection pad if image is smaller than target
        h_c, w_c = gray_crop.shape
        if h_c < crop_size or w_c < crop_size:
            gray_crop = cv2.copyMakeBorder(
                gray_crop,
                0, crop_size - h_c,
                0, crop_size - w_c,
                cv2.BORDER_REFLECT
            )

        gray = gray_crop

        # Step 2: Apply high-pass filter (optional)
        if apply_highpass:
            # Gaussian blur to get low-frequency components
            low_freq = cv2.GaussianBlur(gray, (21, 21), 0)
            # Subtract to get high-frequency components
            gray = cv2.subtract(gray, low_freq)
            # Normalize back to 0-255 range
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to float32 for DFT
        gray_float = np.float32(gray)

        # Compute DFT
        dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Compute magnitude spectrum
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        # Apply log transform
        magnitude_log = 20 * np.log(magnitude + 1)

        # CRITICAL FIX (Dec 12, 2025): Mask DC + Axes BEFORE normalization
        # This removes social media "JPEG Cross" artifact from rectangular borders
        # which creates high-energy white cross 100x stronger than AI artifacts
        rows, cols = magnitude_log.shape
        crow, ccol = rows // 2, cols // 2

        # Mask DC component (center dot) - removes DC bias
        cv2.circle(magnitude_log, (ccol, crow), 5, 0, -1)

        # Mask central axes (the cross) - removes social media border artifacts
        # Horizontal axis (±1 pixel)
        magnitude_log[crow-1:crow+2, :] = 0
        # Vertical axis (±1 pixel)
        magnitude_log[:, ccol-1:ccol+2] = 0

        # Normalize to 0-255
        magnitude_normalized = cv2.normalize(
            magnitude_log, None, 0, 255, cv2.NORM_MINMAX
        )
        fft_image = np.uint8(magnitude_normalized)

        # Analyze patterns with 5-Sigma peak detection (REFINED - Dec 12, 2025)
        # Logic: AI Grid Stars are BRIGHT (survive 5σ threshold)
        #        Real Grain is NOISY but DIMMER (filtered by 5σ)
        #        High peak count (>2000) indicates overwhelmingly textured real image

        # Calculate mean and standard deviation for adaptive thresholding
        mean_val = np.mean(fft_image)
        std_val = np.std(fft_image)

        # 5-Sigma threshold filters out high-entropy grain while preserving AI grid stars
        peak_threshold = mean_val + (5 * std_val)
        peaks = np.sum(fft_image > peak_threshold)

        # Entropy-aware pattern classification
        if peaks > 2000:
            # Overwhelmingly textured - natural high entropy (real photo with grain/noise)
            pattern_type = "Natural/Chaotic (High Entropy)"
        elif peaks > 20:
            # Sparse bright peaks surviving 5σ - high frequency artifacts (AI signature)
            pattern_type = "High Freq Artifacts (Suspected AI)"
        else:
            # Very few bright peaks - clean natural image
            pattern_type = "Natural/Clean"

        # Encode as PNG
        success, png_bytes = cv2.imencode('.png', fft_image)
        if not success:
            raise RuntimeError("Failed to encode FFT spectrum as PNG")

        metrics = {
            'pattern_type': pattern_type,
            'peaks_detected': int(peaks),
            'peak_threshold': float(peak_threshold)
        }

        return png_bytes.tobytes(), metrics

    @staticmethod
    def generate_both(
        image_input: Union[str, Image.Image, np.ndarray],
        ela_quality: int = 90,
        ela_scale: int = 15
    ) -> Tuple[bytes, bytes]:
        """
        Convenience method to generate both ELA and FFT in one call.

        Args:
            image_input: Input image as file path, PIL Image, or numpy array
            ela_quality: JPEG quality for ELA (default: 90)
            ela_scale: Scale factor for ELA visibility (default: 15)

        Returns:
            Tuple of (ela_bytes, fft_bytes) as PNG-encoded bytes

        Example:
            >>> ela, fft = ArtifactGenerator.generate_both('photo.jpg')
            >>> with open('ela.png', 'wb') as f:
            ...     f.write(ela)
            >>> with open('fft.png', 'wb') as f:
            ...     f.write(fft)
        """
        ela_bytes = ArtifactGenerator.generate_ela(
            image_input,
            quality=ela_quality,
            scale_factor=ela_scale
        )
        fft_bytes = ArtifactGenerator.generate_fft(image_input)

        return ela_bytes, fft_bytes


# Convenience functions for direct usage
def generate_ela(
    image_input: Union[str, Image.Image, np.ndarray],
    quality: int = 90,
    scale_factor: int = 15
) -> bytes:
    """
    Generate Error Level Analysis (ELA) map.

    Convenience function that calls ArtifactGenerator.generate_ela().
    See ArtifactGenerator.generate_ela() for detailed documentation.
    """
    return ArtifactGenerator.generate_ela(image_input, quality, scale_factor)


def generate_fft(
    image_input: Union[str, Image.Image, np.ndarray]
) -> bytes:
    """
    Generate Fast Fourier Transform (FFT) magnitude spectrum.

    Convenience function that calls ArtifactGenerator.generate_fft().
    See ArtifactGenerator.generate_fft() for detailed documentation.
    """
    return ArtifactGenerator.generate_fft(image_input)


def generate_both(
    image_input: Union[str, Image.Image, np.ndarray],
    ela_quality: int = 90,
    ela_scale: int = 15
) -> Tuple[bytes, bytes]:
    """
    Generate both ELA and FFT artifacts.

    Convenience function that calls ArtifactGenerator.generate_both().
    Returns tuple of (ela_bytes, fft_bytes).
    """
    return ArtifactGenerator.generate_both(image_input, ela_quality, ela_scale)


if __name__ == "__main__":
    # Example usage and basic testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python forensics.py <image_path>")
        print("Generates ELA and FFT maps for the specified image")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Generating forensic artifacts for: {image_path}")

    try:
        # Generate both artifacts
        ela_bytes, fft_bytes = generate_both(image_path)

        # Save outputs
        ela_output = image_path.rsplit('.', 1)[0] + '_ela.png'
        fft_output = image_path.rsplit('.', 1)[0] + '_fft.png'

        with open(ela_output, 'wb') as f:
            f.write(ela_bytes)
        print(f"✓ ELA map saved to: {ela_output}")

        with open(fft_output, 'wb') as f:
            f.write(fft_bytes)
        print(f"✓ FFT spectrum saved to: {fft_output}")

        print("\nForensic Analysis:")
        print("  - Check ELA map: Uniform rainbow = AI, Dark with edges = Real")
        print("  - Check FFT spectrum: Grid/Cross = AI, Chaotic starburst = Real")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
