from PIL import Image
from typing import List, Dict
import io, base64, pandas as pd
import streamlit as st

from config import MODEL_CONFIGS, get_client_and_name


def extract_score_from_analysis(analysis: str) -> float:
    """Extract the numeric score from the analysis string (e.g., 'Score: 2/10')."""
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


def analyze_single_image(
    image: Image.Image,
    prompts: List[str],
    system_prompt: str,
    model_key: str,
) -> Dict:
    """Analyze a single image with the selected model and return classification + score."""
    client, model_name = get_client_and_name(model_key)

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

        score = extract_score_from_analysis(analysis)
        overall_score = max(overall_score, score)

    classification = "AI Generated" if overall_score > 4 else "Real"

    return {
        "classification": classification,
        "analysis": "\n".join(all_responses),
        "score": overall_score,
    }


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
    )
    return response.choices[0].message.content
