import os
import json
from pathlib import Path
from openai import OpenAI

# Load model configurations from JSON file
def load_model_configs(config_path: str = "models.json") -> dict:
    """
    Load model configurations from JSON file.

    Args:
        config_path: Path to models.json file

    Returns:
        Dictionary of model configurations keyed by model ID
    """
    # Try to load from custom models.json first
    json_path = Path(config_path)

    if not json_path.exists():
        # Fall back to default hardcoded configs if JSON doesn't exist
        return get_default_model_configs()

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Convert from list format to dict format keyed by ID
        model_configs = {}
        for model in data.get("models", []):
            model_id = model.pop("id")  # Remove 'id' from dict and use as key
            model_configs[model_id] = model

        return model_configs

    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Error loading models.json: {e}")
        print("Falling back to default configurations...")
        return get_default_model_configs()


def get_default_model_configs() -> dict:
    """
    Get default hardcoded model configurations.
    Used as fallback if models.json doesn't exist or fails to load.
    """
    return {
    # Local vLLM Models
    "InternVL2_5-8B": {
        "display_name": "InternVL 2.5 8B",
        "base_url": "http://100.64.0.1:8000/v1/",
        "model_name": "InternVL2_5-8B",
        "provider": "vllm",
        "api_key": "dummy"
    },
    "InternVL3_5-8B": {
        "display_name": "InternVL 3.5 8B",
        "base_url": "http://localhost:1234/v1/",
        "model_name": "opengvlab_internvl3_5-8b",
        "provider": "vllm",
        "api_key": "dummy"
    },
    "MiniCPM-V-4_5": {
        "display_name": "MiniCPM-V 4.5",
        "base_url": "http://100.64.0.3:8001/v1/",
        "model_name": "openbmb/MiniCPM-V-4_5",
        "provider": "vllm",
        "api_key": "dummy"
    },
    "Qwen3-VL-32B-Instruct": {
        "display_name": "Qwen3 VL 32B",
        "base_url": "http://100.64.0.3:8006/v1/",
        "model_name": "Qwen3-VL-32B-Instruct",
        "provider": "vllm",
        "api_key": "dummy"
    },

    # Cloud Models (OpenAI)
    "gpt-4o": {
        "display_name": "OpenAI GPT-4o (Cloud)",
        "base_url": "https://api.openai.com/v1/",
        "model_name": "gpt-4o",
        "provider": "openai",
        "api_key": os.getenv("OPENAI_API_KEY", "")
    },
    "gpt-4o-mini": {
        "display_name": "OpenAI GPT-4o Mini (Cloud)",
        "base_url": "https://api.openai.com/v1/",
        "model_name": "gpt-4o-mini",
        "provider": "openai",
        "api_key": os.getenv("OPENAI_API_KEY", "")
    },

    # Cloud Models (Anthropic Claude)
    "claude-3-5-sonnet": {
        "display_name": "Claude 3.5 Sonnet (Cloud)",
        "model_name": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "api_key": os.getenv("ANTHROPIC_API_KEY", "")
    },
    "claude-3-5-haiku": {
        "display_name": "Claude 3.5 Haiku (Cloud)",
        "model_name": "claude-3-5-haiku-20241022",
        "provider": "anthropic",
        "api_key": os.getenv("ANTHROPIC_API_KEY", "")
    },

    # Cloud Models (Google Gemini)
    "gemini-2-0-flash-exp": {
        "display_name": "Gemini 2.0 Flash (Cloud)",
        "model_name": "gemini-2.0-flash-exp",
        "provider": "gemini",
        "api_key": os.getenv("GEMINI_API_KEY", "")
    },
    "gemini-1-5-pro": {
        "display_name": "Gemini 1.5 Pro (Cloud)",
        "model_name": "gemini-1.5-pro-latest",
        "provider": "gemini",
        "api_key": os.getenv("GEMINI_API_KEY", "")
    }
}


# Load configurations from JSON or use defaults
MODEL_CONFIGS = load_model_configs()

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
