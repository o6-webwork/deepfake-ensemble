from openai import OpenAI

# One place to define all models + which vLLM port they live on
# Make sure the base_url ports match your docker-compose
MODEL_CONFIGS = {
    "InternVL2_5-8B": {
        "display_name": "InternVL 2.5 8B",
        "base_url": "http://100.64.0.1:8000/v1/",
        "model_name": "InternVL2_5-8B",
    },
    "InternVL3_5-8B": {
        "display_name": "InternVL 3.5 8B",
        "base_url": "http://localhost:1234/v1/",
        "model_name": "opengvlab_internvl3_5-8b",
    },
    "MiniCPM-V-4_5": {
        "display_name": "MiniCPM-V 4.5",
        "base_url": "http://100.64.0.3:8001/v1/",
        "model_name": "openbmb/MiniCPM-V-4_5",
    },
    "Qwen3-VL-32B-Instruct": {
        "display_name": "Qwen3 VL 32B",
        "base_url": "http://100.64.0.3:8006/v1/",
        "model_name": "Qwen3-VL-32B-Instruct",
    }
}

SYSTEM_PROMPT = """
You are a deepfake detection assistant. You will receive an image or video and an analysis prompt.
After completing the analyses, you will give a final 0–10 score:
0 = certainly real, 10 = certainly fake. Anything above 4 is a cause for concern that its likely AI generated.
Provide clear reasoning for all steps.
"""

PROMPTS = [
    """Assess the media and estimate whether it is more likely to be real, manipulated, or AI-generated.
Explain your decision and rate your confidence as: High, Medium, or Low.
Factors to consider:
- Semantic coherence
- Lighting, textures, artifacts
- Any signs of digital editing
"""
]

# helper to build a client for a given model key
def get_client_and_name(model_key: str):
    conf = MODEL_CONFIGS[model_key]
    client = OpenAI(base_url=conf["base_url"], api_key="dummy")  # vLLM ignores key
    return client, conf["model_name"]
