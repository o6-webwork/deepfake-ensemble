"""
Physics-Based Forensics for Social Media Deepfake Detection

This module implements compression-resistant forensic techniques:
1. Eye Reflection Analysis (IoU) - Detects dissociated highlights in eyes
2. Lighting Consistency (Δθ) - Calculates light direction vector mismatches

These techniques survive compression because they rely on geometric and physical
properties rather than pixel-level noise patterns.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
import math


class EyeReflectionAnalyzer:
    """
    Detects dissociated highlights in eye reflections.

    Principle: In real photos, both eyes reflect the same light source.
    In AI-generated images, reflections often don't match.
    """

    def __init__(self):
        """Initialize face and eye detection cascades."""
        # Using Haar Cascades for robust detection even after compression
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

    def detect_eyes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect eyes in the image.

        Args:
            image: BGR image as numpy array

        Returns:
            List of (x, y, w, h) tuples for each detected eye
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces first
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        eyes = []
        for (fx, fy, fw, fh) in faces:
            # Search for eyes within each face
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            detected_eyes = self.eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )

            for (ex, ey, ew, eh) in detected_eyes:
                # Convert to absolute coordinates
                eyes.append((fx + ex, fy + ey, ew, eh))

        return eyes

    def extract_reflection_region(
        self, image: np.ndarray, eye_region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract the bright reflection region from an eye.

        Args:
            image: BGR image
            eye_region: (x, y, w, h) of eye

        Returns:
            Binary mask of reflection highlights
        """
        x, y, w, h = eye_region
        eye_img = image[y:y+h, x:x+w]

        # Convert to grayscale and enhance contrast
        gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to isolate bright spots
        _, reflection_mask = cv2.threshold(
            gray_eye, 200, 255, cv2.THRESH_BINARY
        )

        return reflection_mask

    def calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two reflection masks.

        Args:
            mask1, mask2: Binary masks of reflections

        Returns:
            IoU score (0.0 to 1.0)
        """
        # Resize masks to same size for comparison
        # Use MIN dimensions to avoid inflating IoU on AI-generated eyes with different reflections
        # Previously used MAX which artificially increased similarity
        h = min(mask1.shape[0], mask2.shape[0])
        w = min(mask1.shape[1], mask2.shape[1])

        mask1_resized = cv2.resize(mask1, (w, h))
        mask2_resized = cv2.resize(mask2, (w, h))

        # Calculate intersection and union
        intersection = np.logical_and(mask1_resized > 0, mask2_resized > 0).sum()
        union = np.logical_or(mask1_resized > 0, mask2_resized > 0).sum()

        if union == 0:
            return 0.0

        return float(intersection) / float(union)

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze eye reflections for consistency.

        Args:
            image_input: Path to image, PIL Image, or numpy array

        Returns:
            {
                "eyes_detected": int,
                "iou_score": float (0.0-1.0, or None if <2 eyes),
                "reflection_consistent": bool,
                "confidence": str ("high", "medium", "low"),
                "verdict": str ("Real" or "AI Generated")
            }
        """
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image = image_input

        # Detect eyes
        eyes = self.detect_eyes(image)

        result = {
            "eyes_detected": len(eyes),
            "iou_score": None,
            "reflection_consistent": None,
            "confidence": "low",
            "verdict": "Inconclusive"
        }

        # Need at least 2 eyes for comparison
        if len(eyes) < 2:
            result["confidence"] = "low"
            result["verdict"] = "Inconclusive"
            return result

        # Extract reflection masks from first two eyes
        reflection1 = self.extract_reflection_region(image, eyes[0])
        reflection2 = self.extract_reflection_region(image, eyes[1])

        # Calculate IoU
        iou = self.calculate_iou(reflection1, reflection2)
        result["iou_score"] = iou

        # Interpretation (conservative thresholds):
        # High IoU (>0.5) = Consistent reflections → Real
        # Medium IoU (0.2-0.5) = Partially consistent → Uncertain
        # Low IoU (<0.2) = Dissociated reflections → AI Generated

        if iou > 0.5:
            result["reflection_consistent"] = True
            result["confidence"] = "high"
            result["verdict"] = "Real"
        elif iou > 0.2:
            result["reflection_consistent"] = None
            result["confidence"] = "medium"
            result["verdict"] = "Uncertain"
        else:
            result["reflection_consistent"] = False
            result["confidence"] = "high"
            result["verdict"] = "AI Generated"

        return result


class LightingConsistencyAnalyzer:
    """
    Analyzes lighting direction consistency between face and background.

    Principle: In real photos, the light source illuminates both face and
    background consistently. In deepfakes, lighting vectors often mismatch.
    """

    def __init__(self):
        """Initialize face detection cascade."""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the primary face in the image.

        Args:
            image: BGR image

        Returns:
            (x, y, w, h) of largest detected face, or None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        if len(faces) == 0:
            return None

        # Return largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)

    def estimate_light_direction(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Estimate light direction from gradient analysis.

        Args:
            image: BGR image region

        Returns:
            (theta_x, theta_y) light direction angles in degrees
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        # Calculate mean gradient direction
        mean_grad_x = np.mean(grad_x)
        mean_grad_y = np.mean(grad_y)

        # Convert to angles
        theta_x = math.degrees(math.atan2(mean_grad_y, mean_grad_x))
        theta_y = math.degrees(math.atan2(mean_grad_x, mean_grad_y))

        return (theta_x, theta_y)

    def calculate_angle_difference(
        self, angles1: Tuple[float, float], angles2: Tuple[float, float]
    ) -> float:
        """
        Calculate angular difference between two lighting directions.

        Args:
            angles1, angles2: (theta_x, theta_y) tuples

        Returns:
            Delta theta (Δθ) in degrees
        """
        # Calculate Euclidean distance in angle space
        dx = angles1[0] - angles2[0]
        dy = angles1[1] - angles2[1]

        # Handle angle wrapping
        dx = min(abs(dx), 360 - abs(dx))
        dy = min(abs(dy), 360 - abs(dy))

        delta_theta = math.sqrt(dx**2 + dy**2)
        return delta_theta

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze lighting consistency between face and background.

        Args:
            image_input: Path to image, PIL Image, or numpy array

        Returns:
            {
                "face_detected": bool,
                "face_light_direction": Tuple[float, float] or None,
                "background_light_direction": Tuple[float, float] or None,
                "delta_theta": float (degrees) or None,
                "lighting_consistent": bool,
                "confidence": str ("high", "medium", "low"),
                "verdict": str ("Real" or "AI Generated")
            }
        """
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image = image_input

        result = {
            "face_detected": False,
            "face_light_direction": None,
            "background_light_direction": None,
            "delta_theta": None,
            "lighting_consistent": None,
            "confidence": "low",
            "verdict": "Inconclusive"
        }

        # Detect face
        face = self.detect_face(image)
        if face is None:
            return result

        result["face_detected"] = True
        x, y, w, h = face

        # Extract face and background regions
        face_region = image[y:y+h, x:x+w]

        # Create background mask (entire image except face)
        background_mask = np.ones(image.shape[:2], dtype=bool)
        background_mask[y:y+h, x:x+w] = False

        # Extract background region (sample from edges)
        bg_sample = image[background_mask]
        if len(bg_sample) < 100:  # Need sufficient background
            return result

        # Estimate light directions
        face_light = self.estimate_light_direction(face_region)

        # Sample background from edges
        edge_width = min(image.shape[0], image.shape[1]) // 4
        bg_top = image[:edge_width, :]
        bg_bottom = image[-edge_width:, :]
        bg_combined = np.vstack([bg_top, bg_bottom])
        background_light = self.estimate_light_direction(bg_combined)

        result["face_light_direction"] = face_light
        result["background_light_direction"] = background_light

        # Calculate angular difference
        delta_theta = self.calculate_angle_difference(face_light, background_light)
        result["delta_theta"] = delta_theta

        # Interpretation:
        # Conservative thresholds:
        # Small Δθ (<25°) = Consistent lighting → Real
        # Medium Δθ (25-60°) = Partially consistent → Uncertain
        # Large Δθ (>60°) = Inconsistent lighting → AI Generated

        if delta_theta < 25:
            result["lighting_consistent"] = True
            result["confidence"] = "high"
            result["verdict"] = "Real"
        elif delta_theta < 60:
            result["lighting_consistent"] = None
            result["confidence"] = "medium"
            result["verdict"] = "Uncertain"
        else:
            result["lighting_consistent"] = False
            result["confidence"] = "high"
            result["verdict"] = "AI Generated"

        return result


class PhysicsForensicsPipeline:
    """
    Physics-based forensics using Eye Reflection IoU only.

    Lighting consistency removed due to false positives on real images
    with complex lighting (multiple sources, outdoor, mixed indoor).
    """

    def __init__(self):
        """Initialize both analyzers."""
        self.eye_analyzer = EyeReflectionAnalyzer()
        self.lighting_analyzer = LightingConsistencyAnalyzer()
        # Cache face cascade to avoid reloading (performance optimization)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Lighting now GATED - only runs when prerequisites met

    def analyze(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Run physics-based forensic analysis (Eye Reflection IoU only).

        Args:
            image_input: Path to image, PIL Image, or numpy array

        Returns:
            {
                "eye_reflection": Dict,
                "lighting_consistency": Dict (kept for backward compatibility, always "N/A"),
                "combined_verdict": str ("Real", "AI Generated", or "Uncertain"),
                "confidence": str ("high", "medium", "low"),
                "explanation": str
            }
        """
        # Convert input to CV2 format for prerequisite checks
        if isinstance(image_input, str):
            image_cv = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            image_cv = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_input

        # Run eye reflection analysis
        eye_result = self.eye_analyzer.analyze(image_input)

        # GATING LOGIC: Only run lighting if prerequisites met
        should_run_lighting = False
        if eye_result["eyes_detected"] >= 2:
            # Check image size
            h, w = image_cv.shape[:2]
            img_area = h * w

            # Detect face for sizing check (using cached cascade)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if len(image_cv.shape) == 3 else image_cv
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

            if len(faces) > 0:
                # Get largest face
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, fw, fh = face
                face_area = fw * fh
                face_pct = (face_area / img_area) * 100

                # Prerequisites:
                # 1. Face occupies > 10% of image
                # 2. Face width > 100 pixels (sufficient resolution)
                # 3. Image resolution > 200x200
                if face_pct > 10 and fw > 100 and min(h, w) > 200:
                    should_run_lighting = True

        # Run lighting analysis only if prerequisites met
        if should_run_lighting:
            lighting_result = self.lighting_analyzer.analyze(image_input)
        else:
            # Prerequisites not met - skip lighting
            lighting_result = {
                "verdict": "N/A",
                "confidence": "low",
                "delta_theta": None,
                "lighting_consistent": None,
                "skip_reason": "Prerequisites not met (face too small or low resolution)"
            }

        # Combine verdicts (with gated lighting)
        verdicts = []
        confidences = []

        # Add eye reflection if available
        if eye_result["verdict"] != "Inconclusive":
            verdicts.append(eye_result["verdict"])
            confidences.append(eye_result["confidence"])

        # Add lighting if it ran (not N/A)
        if lighting_result["verdict"] not in ["N/A", "Inconclusive"]:
            verdicts.append(lighting_result["verdict"])
            confidences.append(lighting_result["confidence"])

        # Determine combined verdict
        if not verdicts:
            combined_verdict = "N/A"
            combined_confidence = "low"
            explanation = "No valid physics features detected"
        elif len(verdicts) == 1:
            # Only one technique ran
            combined_verdict = verdicts[0]
            combined_confidence = confidences[0]
        else:
            # Both techniques ran - check agreement
            if verdicts[0] == verdicts[1]:
                # Both agree
                combined_verdict = verdicts[0]
                combined_confidence = "high" if all(c == "high" for c in confidences) else "medium"
            else:
                # Disagree - be conservative
                combined_verdict = "Uncertain"
                combined_confidence = "low"

        # Generate explanation
        exp_parts = []
        if eye_result["verdict"] != "Inconclusive" and eye_result["iou_score"] is not None:
            exp_parts.append(f"Eye IoU: {eye_result['iou_score']:.2f} ({eye_result['verdict']})")
        if lighting_result["verdict"] not in ["N/A", "Inconclusive"]:
            if lighting_result["delta_theta"] is not None:
                exp_parts.append(f"Lighting Δθ: {lighting_result['delta_theta']:.1f}° ({lighting_result['verdict']})")
        elif "skip_reason" in lighting_result:
            exp_parts.append(f"Lighting: {lighting_result['skip_reason']}")

        explanation = " | ".join(exp_parts) if exp_parts else "No valid features"

        return {
            "eye_reflection": eye_result,
            "lighting_consistency": lighting_result,  # Kept for backward compatibility (always N/A)
            "combined_verdict": combined_verdict,
            "confidence": combined_confidence,
            "explanation": explanation
        }
