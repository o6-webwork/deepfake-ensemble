"""
Texture & Artifact Analysis for Social Media Deepfake Detection

This module implements compression-resistant texture analysis techniques:
1. PLGF (Pattern of Local Gravitational Force) - Robust texture descriptor
2. Frequency Analysis (1D FFT Profile) - Detects checkerboard energy spikes
3. PLADA (Pay Less Attention to Deceptive Artifacts) - Deep learning classifier

These techniques are designed to survive JPEG compression and resizing.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Union, Tuple
import math


class DegradationProfiler:
    """
    Step 0: Estimate how degraded/compressed the image is.

    Helps calibrate other detectors based on compression severity.
    """

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Estimate degradation level of image.

        Returns:
            {
                "format": str,
                "estimated_quality": float (0-100),
                "recompression_hints": int,
                "degradation_level": str ("low", "medium", "high"),
                "explanation": str
            }
        """
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            # Try to get format from file
            from pathlib import Path
            img_format = Path(image_input).suffix.upper().replace(".", "")
        elif isinstance(image_input, Image.Image):
            img_format = image_input.format or "UNKNOWN"
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image = image_input
            img_format = "ARRAY"

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Estimate JPEG quality by analyzing blocking artifacts
        h, w = gray.shape
        block_size = 8

        block_variances = []
        edge_strengths = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_variances.append(np.var(block))

                # Check for blocking edges (JPEG artifacts)
                if j + block_size < w:
                    edge = np.abs(gray[i:i+block_size, j+block_size-1].astype(float) -
                                 gray[i:i+block_size, j+block_size].astype(float))
                    edge_strengths.append(np.mean(edge))

        # High edge strength at block boundaries = heavy JPEG compression
        avg_edge_strength = np.mean(edge_strengths) if edge_strengths else 0
        block_variance = np.std(block_variances) if block_variances else 0

        # Estimate quality (inverse of blocking artifacts)
        blocking_score = avg_edge_strength / (block_variance + 1)
        estimated_quality = max(0, min(100, 100 - blocking_score * 50))

        # Degradation level
        if estimated_quality > 80:
            degradation_level = "low"
        elif estimated_quality > 50:
            degradation_level = "medium"
        else:
            degradation_level = "high"

        # Recompression hints (high blocking = likely recompressed)
        recompression_hints = 1 if blocking_score > 0.5 else 0
        if block_variance < 10:  # Very uniform = heavily compressed
            recompression_hints += 1

        explanation = f"Format: {img_format}, Est. quality: {estimated_quality:.0f}%, Degradation: {degradation_level}"

        return {
            "format": img_format,
            "estimated_quality": estimated_quality,
            "recompression_hints": recompression_hints,
            "degradation_level": degradation_level,
            "blocking_score": blocking_score,
            "explanation": explanation
        }


class NoiseConsistencyAnalyzer:
    """
    Detect local inconsistencies (splicing, copy-paste, inpainting).

    Instead of "is it fake?", answers "where is it inconsistent?"
    """

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze noise pattern consistency across image regions.

        Returns:
            {
                "inconsistency_score": float (0-1),
                "suspicious_regions": int,
                "verdict": str,
                "confidence": str,
                "explanation": str
            }
        """
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image = image_input

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize for efficiency
        h, w = gray.shape
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        h, w = gray.shape

        # Extract high-frequency noise residual
        # Median filter removes content, leaves noise
        median = cv2.medianBlur(gray, 5)
        noise = gray.astype(float) - median.astype(float)

        # Divide into regions and analyze noise statistics
        region_size = 64
        noise_stats = []

        for i in range(0, h - region_size, region_size // 2):
            for j in range(0, w - region_size, region_size // 2):
                region_noise = noise[i:i+region_size, j:j+region_size]

                # Calculate noise characteristics
                noise_std = np.std(region_noise)
                noise_kurtosis = np.mean((region_noise - np.mean(region_noise))**4) / (noise_std**4 + 1e-8)

                noise_stats.append({
                    'std': noise_std,
                    'kurtosis': noise_kurtosis,
                    'position': (i, j)
                })

        if not noise_stats:
            return {
                "inconsistency_score": 0,
                "suspicious_regions": 0,
                "verdict": "Uncertain",
                "confidence": "low",
                "explanation": "Image too small for noise analysis"
            }

        # Compare regions - inconsistent noise = splicing/manipulation
        std_values = [s['std'] for s in noise_stats]
        kurtosis_values = [s['kurtosis'] for s in noise_stats]

        # Coefficient of variation for noise consistency
        std_cv = np.std(std_values) / (np.mean(std_values) + 1e-8)
        kurtosis_cv = np.std(kurtosis_values) / (np.mean(kurtosis_values) + 1e-8)

        # High CV = inconsistent noise = likely manipulation
        inconsistency_score = (std_cv + kurtosis_cv) / 2
        inconsistency_score = min(inconsistency_score, 1.0)

        # Count suspicious regions (outliers)
        mean_std = np.mean(std_values)
        std_threshold = mean_std * 1.5
        suspicious_regions = sum(1 for s in noise_stats if abs(s['std'] - mean_std) > std_threshold)

        # Classification based on inconsistency (not binary fake/real)
        if inconsistency_score > 0.6:
            verdict = "AI Generated"  # High inconsistency = likely synthetic/manipulated
            confidence = "high" if inconsistency_score > 0.8 else "medium"
        elif inconsistency_score < 0.3:
            verdict = "Real"  # Consistent noise = likely authentic
            confidence = "high" if inconsistency_score < 0.15 else "medium"
        else:
            verdict = "Uncertain"
            confidence = "low"

        explanation = f"Inconsistency: {inconsistency_score:.2f}, Suspicious regions: {suspicious_regions}/{len(noise_stats)}"

        return {
            "inconsistency_score": inconsistency_score,
            "suspicious_regions": suspicious_regions,
            "total_regions": len(noise_stats),
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation
        }


class C2PAValidator:
    """
    Check for C2PA (Content Credentials) and verify signature chain.

    If valid → strong evidence of authenticity
    If missing → neutral (social platforms strip metadata)
    """

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Check for C2PA content credentials.

        Returns:
            {
                "has_c2pa": bool,
                "valid_signature": bool or None,
                "credential_type": str or None,
                "verdict": str,
                "explanation": str
            }
        """
        # Load as PIL Image to check metadata
        if isinstance(image_input, str):
            pil_image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input
        else:
            # Convert numpy array to PIL
            if len(image_input.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image_input)

        # Check for C2PA markers in metadata
        # C2PA typically stored in XMP or dedicated segments
        has_c2pa = False
        credential_type = None

        # Check EXIF/metadata
        if hasattr(pil_image, 'info'):
            info = pil_image.info

            # Look for C2PA markers
            for key in info.keys():
                key_lower = str(key).lower()
                if 'c2pa' in key_lower or 'content_credentials' in key_lower or 'provenance' in key_lower:
                    has_c2pa = True
                    credential_type = "detected"
                    break

        # Check EXIF for camera/software tags (weak signal)
        try:
            if hasattr(pil_image, '_getexif') and pil_image._getexif():
                exif = pil_image._getexif()
                if exif:
                    # Check for camera make/model
                    if 271 in exif or 272 in exif:  # Make, Model
                        credential_type = "camera_metadata"
        except:
            pass

        # Verdict based on C2PA presence
        if has_c2pa:
            verdict = "Real"
            explanation = "C2PA credentials detected (strong authenticity signal)"
        elif credential_type == "camera_metadata":
            verdict = "Real"
            explanation = "Camera metadata present (moderate authenticity signal)"
        else:
            verdict = "Uncertain"
            explanation = "No C2PA/credentials (neutral - often stripped by social platforms)"

        return {
            "has_c2pa": has_c2pa,
            "valid_signature": None,  # Would require C2PA library for full validation
            "credential_type": credential_type,
            "verdict": verdict,
            "confidence": "medium" if has_c2pa else "low",
            "explanation": explanation
        }


class DCTAnalyzer:
    """
    Discrete Cosine Transform analyzer for detecting AI-generated artifacts.

    Works on ALL images (faces, landscapes, objects, etc.)
    Analyzes block-wise DCT coefficients to detect unnatural compression patterns.
    """

    def extract_dct_features(self, image: np.ndarray) -> Dict:
        """
        Extract DCT-based features from image.

        Args:
            image: BGR or grayscale image

        Returns:
            Dictionary of DCT features
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize to manageable size
        h, w = gray.shape
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        h, w = gray.shape
        block_size = 8  # Standard DCT block size (like JPEG)

        high_freq_energies = []
        low_freq_energies = []
        total_energies = []

        # Block-wise DCT analysis
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]

                # Apply DCT
                dct_block = cv2.dct(np.float32(block))

                # Separate high and low frequency components
                low_freq = dct_block[0:4, 0:4]  # Top-left quadrant
                high_freq = dct_block[4:8, 4:8]  # Bottom-right quadrant

                # Calculate energies
                low_energy = np.sum(np.abs(low_freq))
                high_energy = np.sum(np.abs(high_freq))
                total_energy = np.sum(np.abs(dct_block))

                low_freq_energies.append(low_energy)
                high_freq_energies.append(high_energy)
                total_energies.append(total_energy)

        # Calculate statistics
        high_freq_ratio = np.mean(high_freq_energies) / (np.mean(total_energies) + 1e-8)
        high_freq_variance = np.std(high_freq_energies)

        # AI images typically have:
        # - Lower high-frequency content (smoother)
        # - More uniform DCT patterns (less variance)

        # Normalize to 0-10 scale (higher = more real-like texture)
        dct_score = (high_freq_ratio * 100 + high_freq_variance / 10)
        dct_score = min(max(dct_score, 0), 10)

        return {
            "high_freq_ratio": high_freq_ratio,
            "high_freq_variance": high_freq_variance,
            "dct_score": dct_score
        }

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze image using DCT.

        Args:
            image_input: Path to image, PIL Image, or numpy array

        Returns:
            Analysis results
        """
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image = image_input

        # Extract features
        features = self.extract_dct_features(image)
        dct_score = features["dct_score"]

        # Classify (conservative thresholds)
        if dct_score > 6.5:
            verdict = "Real"
            confidence = "high" if dct_score > 8.0 else "medium"
        elif dct_score < 3.5:
            verdict = "AI Generated"
            confidence = "high" if dct_score < 2.0 else "medium"
        else:
            verdict = "Uncertain"
            confidence = "low"

        explanation = f"DCT score: {dct_score:.2f}/10 (High-freq ratio: {features['high_freq_ratio']:.3f})"

        return {
            "features": features,
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation
        }


class DWTAnalyzer:
    """
    Discrete Wavelet Transform analyzer using multi-scale decomposition.

    Works on ALL images - detects AI artifacts at multiple scales.
    """

    def extract_dwt_features(self, image: np.ndarray) -> Dict:
        """
        Extract wavelet features using 2D DWT.

        Args:
            image: BGR or grayscale image

        Returns:
            Dictionary of wavelet features
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize for efficiency
        h, w = gray.shape
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        # Normalize
        gray = gray.astype(np.float32) / 255.0

        # Simple 2D Haar Wavelet Transform (without pywt dependency)
        # Using numpy for basic wavelet decomposition
        h, w = gray.shape

        # Ensure even dimensions
        if h % 2 != 0:
            gray = gray[:-1, :]
        if w % 2 != 0:
            gray = gray[:, :-1]

        h, w = gray.shape

        # Level 1 decomposition (horizontal)
        low_h = (gray[:, 0::2] + gray[:, 1::2]) / 2
        high_h = (gray[:, 0::2] - gray[:, 1::2]) / 2

        # Level 1 decomposition (vertical) on both
        LL = (low_h[0::2, :] + low_h[1::2, :]) / 2   # Approximation
        LH = (low_h[0::2, :] - low_h[1::2, :]) / 2   # Horizontal detail
        HL = (high_h[0::2, :] + high_h[1::2, :]) / 2  # Vertical detail
        HH = (high_h[0::2, :] - high_h[1::2, :]) / 2  # Diagonal detail

        # Calculate energies
        detail_energy = (np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)) / 3
        approx_energy = np.sum(LL**2)

        # Calculate variances (texture richness)
        detail_variance = (np.var(LH) + np.var(HL) + np.var(HH)) / 3

        # AI images typically have:
        # - Lower detail energy (smoother)
        # - Lower detail variance (less texture variation)

        # Normalize to 0-10 scale
        wavelet_score = (detail_energy * 50 + detail_variance * 100)
        wavelet_score = min(max(wavelet_score, 0), 10)

        return {
            "detail_energy": detail_energy,
            "approx_energy": approx_energy,
            "detail_variance": detail_variance,
            "wavelet_score": wavelet_score
        }

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze image using DWT.

        Args:
            image_input: Path to image, PIL Image, or numpy array

        Returns:
            Analysis results
        """
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image = image_input

        # Extract features
        features = self.extract_dwt_features(image)
        wavelet_score = features["wavelet_score"]

        # Classify (conservative thresholds)
        if wavelet_score > 6.0:
            verdict = "Real"
            confidence = "high" if wavelet_score > 7.5 else "medium"
        elif wavelet_score < 3.0:
            verdict = "AI Generated"
            confidence = "high" if wavelet_score < 1.5 else "medium"
        else:
            verdict = "Uncertain"
            confidence = "low"

        explanation = f"Wavelet score: {wavelet_score:.2f}/10 (Detail energy: {features['detail_energy']:.4f})"

        return {
            "features": features,
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation
        }


class PLGFAnalyzer:
    """
    Pattern of Local Gravitational Force (PLGF) texture descriptor.

    Models pixels as exerting "gravitational force" on neighbors.
    Distinguishes "smooth" AI gradients from "noisy" real sensor gradients.
    """

    def __init__(self, radius: int = 3):
        """
        Initialize PLGF analyzer.

        Args:
            radius: Neighborhood radius for force calculation
        """
        self.radius = radius

    def calculate_gravitational_force(
        self, image: np.ndarray, center: Tuple[int, int]
    ) -> np.ndarray:
        """
        Calculate gravitational force vector at a pixel.

        Args:
            image: Grayscale image
            center: (y, x) coordinates of center pixel

        Returns:
            Force vector [fx, fy]
        """
        cy, cx = center
        h, w = image.shape

        force_x = 0.0
        force_y = 0.0

        # Calculate force from all neighbors within radius
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                if dy == 0 and dx == 0:
                    continue

                ny, nx = cy + dy, cx + dx

                # Check bounds
                if 0 <= ny < h and 0 <= nx < w:
                    # "Mass" is pixel intensity difference
                    mass = abs(float(image[ny, nx]) - float(image[cy, cx]))

                    # Distance
                    distance = math.sqrt(dx**2 + dy**2)

                    if distance > 0:
                        # Gravitational force: F = m / d^2
                        force_magnitude = mass / (distance ** 2)

                        # Direction
                        force_x += force_magnitude * (dx / distance)
                        force_y += force_magnitude * (dy / distance)

        return np.array([force_x, force_y])

    def extract_plgf_features(self, image: np.ndarray) -> Dict:
        """
        Extract PLGF features from image.

        Args:
            image: BGR image

        Returns:
            {
                "force_magnitude_mean": float,
                "force_magnitude_std": float,
                "force_direction_variance": float,
                "texture_score": float (0-10)
            }
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Sample pixels (grid sampling for efficiency)
        h, w = gray.shape
        step = max(h, w) // 50  # Sample ~50x50 grid
        step = max(step, 1)

        force_magnitudes = []
        force_directions = []

        for y in range(self.radius, h - self.radius, step):
            for x in range(self.radius, w - self.radius, step):
                force = self.calculate_gravitational_force(gray, (y, x))
                magnitude = np.linalg.norm(force)
                direction = math.atan2(force[1], force[0])

                force_magnitudes.append(magnitude)
                force_directions.append(direction)

        # Calculate statistics
        force_mag_mean = np.mean(force_magnitudes)
        force_mag_std = np.std(force_magnitudes)
        force_dir_variance = np.var(force_directions)

        # Interpretation:
        # AI-generated images: Low variance (smooth gradients)
        # Real photos: High variance (noisy sensor gradients)

        # Normalize texture score (0-10)
        # Higher score = more likely AI (smooth)
        texture_score = max(0, min(10, 10 - (force_mag_std / force_mag_mean * 10)))

        return {
            "force_magnitude_mean": force_mag_mean,
            "force_magnitude_std": force_mag_std,
            "force_direction_variance": force_dir_variance,
            "texture_score": texture_score
        }

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze image texture using PLGF.

        Args:
            image_input: Path to image, PIL Image, or numpy array

        Returns:
            {
                "features": Dict,
                "verdict": str ("Real" or "AI Generated"),
                "confidence": str ("high", "medium", "low"),
                "explanation": str
            }
        """
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image = image_input

        # Extract features
        features = self.extract_plgf_features(image)
        texture_score = features["texture_score"]

        # Classify based on texture score
        if texture_score > 6:
            verdict = "AI Generated"
            confidence = "high" if texture_score > 7.5 else "medium"
        elif texture_score < 4:
            verdict = "Real"
            confidence = "high" if texture_score < 2.5 else "medium"
        else:
            verdict = "Uncertain"
            confidence = "low"

        explanation = (
            f"Texture smoothness score: {texture_score:.1f}/10 "
            f"(Force std: {features['force_magnitude_std']:.2f})"
        )

        return {
            "features": features,
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation
        }


class FrequencyAnalyzer:
    """
    1D FFT Profile Analysis.

    Detects checkerboard energy spikes in frequency domain,
    a hallmark of GAN upsampling artifacts.
    """

    def __init__(self):
        """Initialize frequency analyzer."""
        pass

    def compute_1d_fft_profile(self, image: np.ndarray) -> np.ndarray:
        """
        Compute 1D FFT power spectrum profile.

        Args:
            image: Grayscale image

        Returns:
            1D array of frequency magnitudes
        """
        # Convert to float32
        gray_float = np.float32(image)

        # Compute 2D FFT
        dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Compute magnitude spectrum
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        # Convert to log scale
        magnitude_log = 20 * np.log(magnitude + 1)

        # Extract 1D profile by radial averaging
        h, w = magnitude_log.shape
        center_y, center_x = h // 2, w // 2

        # Calculate radial distances
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)

        # Radial averaging
        max_r = min(center_x, center_y)
        radial_profile = np.zeros(max_r)

        for radius in range(max_r):
            mask = (r == radius)
            if mask.any():
                radial_profile[radius] = magnitude_log[mask].mean()

        return radial_profile

    def detect_periodic_spikes(self, profile: np.ndarray, threshold: float = 1.5) -> Dict:
        """
        Detect periodic spikes in frequency profile.

        Args:
            profile: 1D frequency profile
            threshold: Spike detection threshold (std deviations)

        Returns:
            {
                "spike_count": int,
                "spike_strength": float,
                "has_checkerboard": bool
            }
        """
        # Calculate baseline (smoothed profile) using simple moving average
        kernel_size = 11
        kernel = np.ones(kernel_size) / kernel_size
        baseline = np.convolve(profile, kernel, mode='same')

        # Detect deviations from baseline
        deviations = profile - baseline

        # Find spikes (values exceeding threshold * std)
        std_dev = np.std(deviations)
        spike_mask = deviations > (threshold * std_dev)

        spike_count = np.sum(spike_mask)
        spike_strength = np.mean(deviations[spike_mask]) if spike_count > 0 else 0.0

        # Checkerboard pattern typically shows strong mid-frequency spikes
        mid_freq_start = len(profile) // 4
        mid_freq_end = len(profile) // 2
        mid_freq_spikes = np.sum(spike_mask[mid_freq_start:mid_freq_end])

        # Conservative threshold to reduce JPEG compression false positives
        has_checkerboard = mid_freq_spikes > 6 and spike_strength > (std_dev * 1.5)

        return {
            "spike_count": int(spike_count),
            "spike_strength": float(spike_strength),
            "has_checkerboard": bool(has_checkerboard)
        }

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze image frequency characteristics.

        Args:
            image_input: Path to image, PIL Image, or numpy array

        Returns:
            {
                "spike_count": int,
                "spike_strength": float,
                "has_checkerboard": bool,
                "verdict": str ("Real" or "AI Generated"),
                "confidence": str ("high", "medium", "low"),
                "explanation": str
            }
        """
        # Load image
        if isinstance(image_input, str):
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_input, Image.Image):
            gray = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2GRAY)
        else:
            if len(image_input.shape) == 3:
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_input

        # Compute 1D FFT profile
        profile = self.compute_1d_fft_profile(gray)

        # Detect spikes
        spike_info = self.detect_periodic_spikes(profile)

        # Classify
        if spike_info["has_checkerboard"]:
            verdict = "AI Generated"
            confidence = "high"
            explanation = (
                f"Detected checkerboard pattern in frequency domain "
                f"({spike_info['spike_count']} spikes, strength: {spike_info['spike_strength']:.2f})"
            )
        elif spike_info["spike_count"] > 5:
            verdict = "AI Generated"
            confidence = "medium"
            explanation = (
                f"Detected {spike_info['spike_count']} frequency spikes, "
                f"possible GAN artifacts"
            )
        else:
            verdict = "Real"
            confidence = "medium"
            explanation = (
                f"Natural frequency distribution detected "
                f"({spike_info['spike_count']} minor spikes)"
            )

        return {
            **spike_info,
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation
        }


class PLADAClassifier:
    """
    PLADA (Pay Less Attention to Deceptive Artifacts) classifier.

    Uses a simplified heuristic-based approach to ignore JPEG compression
    artifacts while focusing on AI generation clues.

    Note: Full PLADA requires a trained neural network. This implementation
    uses a heuristic approximation based on the paper's principles.
    """

    def __init__(self):
        """Initialize PLADA classifier."""
        pass

    def simulate_social_media_compression(
        self, image: np.ndarray, quality: int = 60
    ) -> np.ndarray:
        """
        Simulate social media compression (blur + resize + JPEG).

        Args:
            image: BGR image
            quality: JPEG quality (default: 60 for social media)

        Returns:
            Compressed image
        """
        # Slight blur
        blurred = cv2.GaussianBlur(image, (3, 3), 0.5)

        # Resize down and up (simulating social media resize)
        h, w = blurred.shape[:2]
        small = cv2.resize(blurred, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        resized = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # JPEG compression
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, compressed_bytes = cv2.imencode('.jpg', resized, encode_params)
        compressed = cv2.imdecode(compressed_bytes, cv2.IMREAD_COLOR)

        return compressed

    def extract_block_effect_features(self, image: np.ndarray) -> float:
        """
        Measure JPEG blocking artifacts (to ignore them).

        Args:
            image: Grayscale image

        Returns:
            Blocking score (0-1)
        """
        # Calculate gradients at 8x8 block boundaries
        h, w = image.shape

        vertical_blocks = []
        horizontal_blocks = []

        # Check vertical block boundaries
        for x in range(8, w, 8):
            if x < w - 1:
                diff = np.abs(image[:, x].astype(float) - image[:, x-1].astype(float))
                vertical_blocks.append(np.mean(diff))

        # Check horizontal block boundaries
        for y in range(8, h, 8):
            if y < h - 1:
                diff = np.abs(image[y, :].astype(float) - image[y-1, :].astype(float))
                horizontal_blocks.append(np.mean(diff))

        if not vertical_blocks and not horizontal_blocks:
            return 0.0

        # Average blocking score
        all_blocks = vertical_blocks + horizontal_blocks
        blocking_score = np.mean(all_blocks) / 255.0

        return min(blocking_score, 1.0)

    def extract_ai_features(self, image: np.ndarray) -> Dict:
        """
        Extract features specific to AI generation (not compression).

        Args:
            image: BGR image

        Returns:
            {
                "color_variance": float,
                "edge_smoothness": float,
                "texture_uniformity": float
            }
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Color variance (AI tends to have lower variance)
        color_variance = np.std(image) / 255.0

        # Edge smoothness (AI edges are smoother)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        edge_smoothness = 1.0 - edge_density

        # Texture uniformity (AI is more uniform)
        texture_uniformity = 1.0 - (np.std(gray) / np.mean(gray))

        return {
            "color_variance": color_variance,
            "edge_smoothness": edge_smoothness,
            "texture_uniformity": texture_uniformity
        }

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Classify image using PLADA approach.

        Args:
            image_input: Path to image, PIL Image, or numpy array

        Returns:
            {
                "blocking_score": float,
                "ai_features": Dict,
                "authenticity_score": float (0-1, 0=fake, 1=real),
                "verdict": str ("Real" or "AI Generated"),
                "confidence": str ("high", "medium", "low"),
                "explanation": str
            }
        """
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image = image_input

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract blocking artifacts (to ignore)
        blocking_score = self.extract_block_effect_features(gray)

        # Extract AI-specific features
        ai_features = self.extract_ai_features(image)

        # Calculate authenticity score
        # Down-weight blocking artifacts, focus on AI features
        ai_score = (
            ai_features["edge_smoothness"] * 0.4 +
            ai_features["texture_uniformity"] * 0.4 +
            (1.0 - ai_features["color_variance"]) * 0.2
        )

        # Authenticity score (inverse of AI score, adjusted for blocking)
        authenticity_score = 1.0 - ai_score

        # Adjust for heavy compression (less confident)
        if blocking_score > 0.3:
            confidence_penalty = 0.2
        else:
            confidence_penalty = 0.0

        # Classify (conservative thresholds to reduce false positives)
        if authenticity_score < 0.25:
            verdict = "AI Generated"
            confidence = "high" if authenticity_score < 0.15 else "medium"
        elif authenticity_score > 0.65:
            verdict = "Real"
            confidence = "high" if authenticity_score > 0.75 else "medium"
        else:
            verdict = "Uncertain"
            confidence = "low"

        # Lower confidence if heavily compressed
        if blocking_score > 0.3 and confidence == "high":
            confidence = "medium"

        explanation = (
            f"Authenticity score: {authenticity_score:.2f} "
            f"(Edge smoothness: {ai_features['edge_smoothness']:.2f}, "
            f"Texture uniformity: {ai_features['texture_uniformity']:.2f}, "
            f"JPEG blocking: {blocking_score:.2f})"
        )

        return {
            "blocking_score": blocking_score,
            "ai_features": ai_features,
            "authenticity_score": authenticity_score,
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation
        }


class EdgeCaseDetector:
    """
    Detect edge cases that require special handling:
    - Screenshots (UI elements, moiré patterns)
    - Heavy beauty filters (over-smoothed skin)
    - Multiple re-compressions (repost artifacts)
    - Screen captures from videos
    """

    def detect_screenshot_artifacts(self, image: np.ndarray) -> Dict:
        """
        Detect screenshot/screen-capture patterns.

        Args:
            image: BGR image

        Returns:
            Detection results
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape

        # Check for UI element patterns (horizontal/vertical lines at edges)
        edge_top = gray[0:int(h*0.05), :]
        edge_bottom = gray[int(h*0.95):, :]

        # High contrast edges suggest UI elements
        top_variance = np.var(edge_top)
        bottom_variance = np.var(edge_bottom)

        # Check for moiré patterns (common in screenshots of screens)
        # Use FFT to detect periodic patterns
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Look for strong periodic peaks (moiré)
        magnitude[h//2-5:h//2+5, w//2-5:w//2+5] = 0  # Zero out DC component
        peak_ratio = np.max(magnitude) / np.mean(magnitude)

        is_screenshot = (peak_ratio > 100 or top_variance > 5000 or bottom_variance > 5000)

        return {
            "is_screenshot": is_screenshot,
            "peak_ratio": peak_ratio,
            "edge_variance": max(top_variance, bottom_variance),
            "confidence": "high" if is_screenshot else "low"
        }

    def detect_beauty_filter(self, image: np.ndarray) -> Dict:
        """
        Detect heavy beauty filter application (over-smoothed skin).

        Args:
            image: BGR image

        Returns:
            Detection results
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Beauty filters create unnaturally smooth skin
        # Calculate local variance in patches
        h, w = l_channel.shape
        patch_size = 32
        variances = []

        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = l_channel[y:y+patch_size, x:x+patch_size]
                variances.append(np.var(patch))

        avg_variance = np.mean(variances)
        min_variance = np.min(variances)

        # Very low variance suggests aggressive smoothing
        has_beauty_filter = (avg_variance < 50 and min_variance < 10)

        return {
            "has_beauty_filter": has_beauty_filter,
            "avg_variance": avg_variance,
            "min_variance": min_variance,
            "confidence": "medium" if has_beauty_filter else "low"
        }

    def detect_multiple_compressions(self, image: np.ndarray) -> Dict:
        """
        Detect signs of multiple JPEG re-compressions (reposts).

        Args:
            image: BGR image

        Returns:
            Detection results
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Multiple compressions accumulate blocking artifacts
        # Check for 8x8 JPEG block boundaries
        h, w = gray.shape

        # Sample block boundaries
        boundary_diffs = []
        for y in range(8, h-8, 8):
            for x in range(8, w-8, 8):
                # Compare pixel values across block boundary
                diff_h = abs(int(gray[y, x]) - int(gray[y-1, x]))
                diff_v = abs(int(gray[y, x]) - int(gray[y, x-1]))
                boundary_diffs.append(max(diff_h, diff_v))

        avg_boundary_diff = np.mean(boundary_diffs) if boundary_diffs else 0

        # Strong blocking suggests multiple compressions
        multiple_compressions = avg_boundary_diff > 15

        return {
            "multiple_compressions": multiple_compressions,
            "avg_boundary_diff": avg_boundary_diff,
            "confidence": "medium" if multiple_compressions else "low"
        }

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Run full edge case detection.

        Args:
            image_input: Image to analyze

        Returns:
            Edge case detection results
        """
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image = image_input

        screenshot = self.detect_screenshot_artifacts(image)
        beauty = self.detect_beauty_filter(image)
        repost = self.detect_multiple_compressions(image)

        # Determine if any edge case detected
        edge_case_flags = []
        if screenshot["is_screenshot"]:
            edge_case_flags.append("screenshot")
        if beauty["has_beauty_filter"]:
            edge_case_flags.append("beauty_filter")
        if repost["multiple_compressions"]:
            edge_case_flags.append("multiple_reposts")

        return {
            "screenshot": screenshot,
            "beauty_filter": beauty,
            "multiple_compressions": repost,
            "edge_cases_detected": edge_case_flags,
            "has_edge_case": len(edge_case_flags) > 0,
            "warning": (
                f"Edge cases detected: {', '.join(edge_case_flags)}"
                if edge_case_flags
                else "No edge cases detected"
            )
        }


class TextureForensicsPipeline:
    """
    Combined pipeline for texture and artifact analysis.
    Aggregates results from DCT, DWT, PLGF, Frequency Analysis, and PLADA.

    Works on ALL image types - faces, landscapes, objects, etc.
    """

    def __init__(self):
        """Initialize all analyzers."""
        # Step 0: Degradation profiling
        self.degradation_profiler = DegradationProfiler()

        # Edge case detection
        self.edge_case_detector = EdgeCaseDetector()

        # Forensic inconsistency detection
        self.noise_analyzer = NoiseConsistencyAnalyzer()
        self.c2pa_validator = C2PAValidator()

        # Frequency/texture analysis
        self.dct_analyzer = DCTAnalyzer()
        self.dwt_analyzer = DWTAnalyzer()
        self.plgf_analyzer = PLGFAnalyzer()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.plada_classifier = PLADAClassifier()

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Run full texture and artifact analysis.

        Args:
            image_input: Path to image, PIL Image, or numpy array

        Returns:
            {
                "plgf": Dict,
                "frequency": Dict,
                "plada": Dict,
                "combined_verdict": str ("Real", "AI Generated", or "Uncertain"),
                "confidence": str ("high", "medium", "low"),
                "explanation": str
            }
        """
        # Step 0: Degradation profiling (understand compression level)
        degradation = self.degradation_profiler.analyze(image_input)

        # Step 0.5: Edge case detection (screenshots, beauty filters, reposts)
        edge_cases = self.edge_case_detector.analyze(image_input)

        # Step 1: Provenance check (C2PA)
        c2pa_result = self.c2pa_validator.analyze(image_input)

        # Step 2: Forensic inconsistency detection
        noise_result = self.noise_analyzer.analyze(image_input)

        # Step 3: Run all frequency/texture analyses
        dct_result = self.dct_analyzer.analyze(image_input)
        dwt_result = self.dwt_analyzer.analyze(image_input)
        plgf_result = self.plgf_analyzer.analyze(image_input)
        frequency_result = self.frequency_analyzer.analyze(image_input)
        plada_result = self.plada_classifier.analyze(image_input)

        # C2PA override: If valid credentials found, strong authenticity signal
        if c2pa_result["has_c2pa"]:
            # C2PA found = very likely real, but still check for inconsistencies
            c2pa_weight = 2  # Double weight for C2PA
        else:
            c2pa_weight = 0  # Neutral when missing

        # Combine verdicts using weighted voting
        # NOTE: FFT demoted to "advisory only" - too fragile (triggers on textures, WebP artifacts)
        verdicts = {
            "noise": noise_result["verdict"],  # Inconsistency detection
            "dct": dct_result["verdict"],
            "dwt": dwt_result["verdict"],
            "plgf": plgf_result["verdict"],
            # "frequency": frequency_result["verdict"],  # REMOVED from voting - advisory only
            "plada": plada_result["verdict"]
        }

        confidences = {
            "noise": noise_result["confidence"],
            "dct": dct_result["confidence"],
            "dwt": dwt_result["confidence"],
            "plgf": plgf_result["confidence"],
            # "frequency": frequency_result["confidence"],  # REMOVED from voting
            "plada": plada_result["confidence"]
        }

        # FFT is advisory only (reported but doesn't vote)
        fft_advisory = frequency_result["verdict"]

        # Count votes
        from collections import Counter
        verdict_counts = Counter(verdicts.values())

        # Conservative voting: require stronger evidence for "AI Generated"
        ai_votes = sum(1 for v in verdicts.values() if v == "AI Generated")
        real_votes = sum(1 for v in verdicts.values() if v == "Real")
        uncertain_votes = sum(1 for v in verdicts.values() if v == "Uncertain")

        # Apply C2PA weight
        if c2pa_result["verdict"] == "Real":
            real_votes += c2pa_weight

        # 3-CATEGORY DECISION RULE
        # Categories: "Real" | "AI Manipulated" | "AI Generated" | "Uncertain"
        #
        # Key: Noise inconsistency differentiates Manipulated vs Generated

        # Check for manipulation (splicing, inpainting, face swaps)
        noise_inconsistency = noise_result["inconsistency_score"]
        suspicious_regions = noise_result["suspicious_regions"]

        # Degradation-adaptive thresholds: adjust for compression artifacts
        # Compressed images have higher baseline noise inconsistency
        degradation_factor = {
            "low": 1.0,     # Clean images: use base thresholds
            "medium": 1.2,  # Moderate compression: +20% tolerance
            "high": 1.5     # Heavy compression: +50% tolerance
        }.get(degradation["degradation_level"], 1.0)

        # Scale thresholds for compression tolerance
        threshold_ai_generated = 0.4 * degradation_factor  # Was 0.4, now 0.4-0.6
        threshold_real = 0.5 * degradation_factor          # Was 0.5, now 0.5-0.75
        threshold_manipulation_low = 0.5 * degradation_factor  # Was 0.5

        # RULE 1: AI Manipulated (editing/splicing detected)
        if noise_inconsistency > 0.6 and suspicious_regions >= 2:
            # High noise variance + suspicious regions = manipulation
            combined_verdict = "AI Manipulated"
            combined_confidence = "high" if suspicious_regions >= 4 else "medium"

        # RULE 2: AI Generated (fully synthetic)
        elif ai_votes >= 3 and noise_inconsistency < threshold_ai_generated:
            # Multiple AI detectors + uniform noise = fully synthetic
            combined_verdict = "AI Generated"
            combined_confidence = "high" if ai_votes >= 4 else "medium"

        # RULE 3: Real (authentic, unedited)
        elif real_votes >= 3 and noise_inconsistency < threshold_real:
            # Multiple Real votes + low-moderate noise inconsistency
            combined_verdict = "Real"
            combined_confidence = "high" if real_votes >= 4 else "medium"

        # RULE 4: Check for manipulation with fewer signals
        elif ai_votes >= 2 and noise_inconsistency > threshold_manipulation_low and suspicious_regions >= 1:
            # Some AI signals + noise inconsistency = likely manipulated
            combined_verdict = "AI Manipulated"
            combined_confidence = "low"

        # RULE 5: Uncertain (mixed or insufficient evidence)
        else:
            combined_verdict = "Uncertain"
            combined_confidence = "low"

        # Adjust confidence based on degradation level
        if degradation["degradation_level"] == "high" and combined_confidence == "high":
            combined_confidence = "medium"  # Lower confidence on heavily compressed images

        # Adjust confidence based on edge cases
        if edge_cases["has_edge_case"]:
            # Screenshots, beauty filters, and reposts can confuse forensic tests
            if combined_confidence == "high":
                combined_confidence = "medium"
            # Don't lower medium to low - edge cases are informative, not disqualifying

        # Generate explanation
        explanations = [
            f"Degradation: {degradation['explanation']}",
            f"Edge cases: {edge_cases['warning']}",
            f"C2PA: {c2pa_result['explanation']}",
            f"Noise: {noise_result['explanation']}",
            f"DCT: {dct_result['explanation']}",
            f"DWT: {dwt_result['explanation']}",
            f"PLGF: {plgf_result['explanation']}",
            f"FFT (advisory): {frequency_result['explanation']}",  # Advisory only
            f"PLADA: {plada_result['explanation']}"
        ]

        explanation = " | ".join(explanations)

        # Add note if FFT disagrees with consensus
        if fft_advisory == "AI Generated" and combined_verdict != "AI Generated":
            explanation += " | Note: FFT flagged but is advisory-only (triggers on textures/WebP)"

        return {
            "degradation": degradation,
            "edge_cases": edge_cases,
            "c2pa": c2pa_result,
            "noise": noise_result,
            "dct": dct_result,
            "dwt": dwt_result,
            "plgf": plgf_result,
            "frequency": frequency_result,
            "plada": plada_result,
            "combined_verdict": combined_verdict,
            "confidence": combined_confidence,
            "explanation": explanation
        }
