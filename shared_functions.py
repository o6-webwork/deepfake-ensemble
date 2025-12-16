from PIL import Image
from typing import List, Dict
import io, base64, pandas as pd
import streamlit as st

from config import MODEL_CONFIGS, get_client_and_name
from detector import OSINTDetector


def extract_verdict_from_detector_output(result: Dict) -> Dict:
    """
    Extract evaluation-compatible verdict from OSINTDetector output.

    Args:
        result: Output from detector.detect() with keys:
            - tier: str ("Authentic" / "Suspicious" / "Deepfake")
            - confidence: float (0.0-1.0, probability of being fake)
            - reasoning: str (Stage 2 VLM analysis)
            - verdict: dict with token, confidence, raw_logits

    Returns:
        {
            "classification": str,  # "Real" or "AI Generated" for metrics
            "analysis": str,       # Full reasoning text
            "confidence": float,   # 0.0-1.0 confidence in the classification
            "tier": str,          # "Authentic" / "Suspicious" / "Deepfake"
            "verdict_token": str  # "A" or "B" from MCQ
        }
    """
    # Extract verdict information
    confidence_fake = result.get('confidence', 0.5)
    tier = result.get('tier', 'Suspicious')
    reasoning = result.get('reasoning', 'No analysis available')
    verdict_info = result.get('verdict', {})
    verdict_token = verdict_info.get('token', 'B' if confidence_fake >= 0.5 else 'A')

    # Determine classification (binary for metrics calculation)
    # Use tier-based logic for consistency:
    # - Authentic (confidence_fake < 0.50) → "Real"
    # - Suspicious/Deepfake (confidence_fake >= 0.50) → "AI Generated"
    if tier == "Authentic":
        classification = "Real"
    else:  # Suspicious or Deepfake
        classification = "AI Generated"

    return {
        "classification": classification,
        "analysis": reasoning,
        "confidence": confidence_fake,
        "tier": tier,
        "verdict_token": verdict_token
    }


def analyze_single_image(
    image: Image.Image,
    model_config: Dict,
    context: str = "auto",
    watermark_mode: str = "ignore",
    detection_mode: str = "spai_assisted",
    spai_max_size: int = 1280,
    spai_overlay_alpha: float = 0.6,
    spai_detector=None
) -> Dict:
    """
    Analyze a single image using the OSINT deepfake detector with SPAI.

    Args:
        image: PIL Image to analyze
        model_config: Model configuration dict from MODEL_CONFIGS
        context: OSINT context ("auto", "military", "disaster", "propaganda")
        watermark_mode: "ignore" or "analyze"
        detection_mode: "spai_standalone" or "spai_assisted"
        spai_max_size: Maximum resolution for SPAI analysis (512-2048 or None)
        spai_overlay_alpha: Transparency for heatmap blending (0.0-1.0)
        spai_detector: Pre-loaded SPAIDetector instance (for caching)

    Returns:
        {
            "classification": str,  # "Real" or "AI Generated"
            "analysis": str,       # Full reasoning
            "confidence": float,   # 0.0-1.0
            "tier": str,          # "Authentic" / "Suspicious" / "Deepfake"
            "verdict_token": str  # "A" or "B"
        }
    """
    # Convert PIL Image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    # Initialize detector with SPAI configuration
    detector = OSINTDetector(
        base_url=model_config.get('base_url', ''),
        model_name=model_config.get('model_name', ''),
        api_key=model_config.get('api_key', 'dummy'),
        context=context,
        watermark_mode=watermark_mode,
        provider=model_config.get('provider', 'vllm'),
        detection_mode=detection_mode,
        spai_detector=spai_detector,  # Pass cached SPAI detector
        spai_max_size=spai_max_size,
        spai_overlay_alpha=spai_overlay_alpha
    )

    # Run detection
    result = detector.detect(
        image_bytes=image_bytes,
        debug=False
    )

    # Convert to evaluation format
    return extract_verdict_from_detector_output(result)


def load_ground_truth(csv_file) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    if not {"filename", "label"}.issubset(df.columns):
        raise ValueError("CSV must have 'filename' and 'label' columns")
    valid_labels = {"Real", "AI Generated"}
    invalid = df[~df["label"].isin(valid_labels)]
    if not invalid.empty:
        raise ValueError(f"Invalid labels: {invalid['label'].unique()}")
    return df


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    y_true_bin = [1 if l == "AI Generated" else 0 for l in y_true]
    y_pred_bin = [1 if l == "AI Generated" else 0 for l in y_pred]
    tp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 0)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def display_confusion_matrix(metrics: Dict[str, float]):
    cm_df = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        columns=["Predicted Real", "Predicted AI"],
        index=["Actual Real", "Actual AI"],
    )
    st.write("### Confusion Matrix")
    st.dataframe(cm_df)
    total = sum([metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]])
    cm_pct = cm_df / total * 100
    st.write("### Confusion Matrix (%)")
    st.dataframe(cm_pct.round(2))


def chat_with_model(messages, system_prompt, model_key: str):
    client, model_name = get_client_and_name(model_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.7,  # Allow creativity for chat interactions
    )
    return response.choices[0].message.content
