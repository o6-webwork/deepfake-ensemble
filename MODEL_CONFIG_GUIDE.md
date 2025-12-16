# Model Configuration Guide (models.json)

**Date:** December 15, 2025
**Feature:** JSON-based model configuration system

---

## Overview

The OSINT Deepfake Detector now supports dynamic model configuration via a JSON file. This allows you to:

1. **Add custom models** without modifying code
2. **Manage API keys** in one centralized file
3. **Share configurations** across team members
4. **Support future models** not yet in the system

---

## Quick Start

### Step 1: Download Template

1. Open the app at `http://localhost:8501`
2. Expand "⚙️ Model Configuration" section
3. Click "📥 Download Template" button
4. Save as `models.json`

**Or** copy the example file:
```bash
cp models.json.example models.json
```

### Step 2: Edit Configuration

Open `models.json` in a text editor and update:

```json
{
  "models": [
    {
      "id": "gpt-4o-cloud",
      "display_name": "OpenAI GPT-4o (Cloud)",
      "base_url": "https://api.openai.com/v1/",
      "model_name": "gpt-4o",
      "provider": "openai",
      "api_key": "sk-YOUR_ACTUAL_API_KEY_HERE"  ← Replace this!
    }
  ]
}
```

### Step 3: Upload Configuration

**Option A: Via UI (Recommended)**
1. Expand "⚙️ Model Configuration"
2. Click "Upload models.json"
3. Select your edited `models.json` file
4. Click "🔄 Reload Models" or refresh page

**Option B: Direct File Placement**
1. Place `models.json` in project root directory
2. Restart Docker container:
   ```bash
   docker compose down
   docker compose up -d
   ```

---

## Configuration Format

### Complete Example

```json
{
  "models": [
    {
      "id": "unique-model-id",
      "display_name": "Model Name (Shown in UI)",
      "base_url": "http://localhost:8000/v1/",
      "model_name": "actual/model/identifier",
      "provider": "vllm",
      "api_key": "dummy"
    }
  ]
}
```

### Field Descriptions

| Field | Required | Description | Example Values |
|-------|----------|-------------|----------------|
| `id` | ✅ Yes | Unique identifier (internal use) | `"qwen3-vl-32b-local"`, `"gpt-4o-cloud"` |
| `display_name` | ✅ Yes | Name shown in UI dropdown | `"Qwen3 VL 32B (Local)"`, `"OpenAI GPT-4o"` |
| `base_url` | Conditional | API endpoint URL | `"http://localhost:8000/v1/"`, `"https://api.openai.com/v1/"` |
| `model_name` | ✅ Yes | Model identifier used by API | `"Qwen3-VL-32B-Instruct"`, `"gpt-4o"` |
| `provider` | ✅ Yes | Provider type | `"vllm"`, `"openai"`, `"anthropic"`, `"gemini"` |
| `api_key` | ✅ Yes | API authentication key | `"dummy"` (local), `"sk-..."` (cloud) |

**Notes:**
- `base_url` is required for `vllm` and `openai` providers
- `base_url` should be empty (`""`) for `anthropic` and `gemini` providers
- `api_key` should be `"dummy"` for local vLLM models

---

## Provider-Specific Configuration

### 1. Local vLLM Models

```json
{
  "id": "my-local-model",
  "display_name": "My Local Model",
  "base_url": "http://localhost:8000/v1/",
  "model_name": "path/to/model",
  "provider": "vllm",
  "api_key": "dummy"
}
```

**Requirements:**
- vLLM server running locally
- `base_url` points to vLLM endpoint
- `api_key` set to `"dummy"` (vLLM ignores it)

### 2. OpenAI Models

```json
{
  "id": "gpt-4o",
  "display_name": "OpenAI GPT-4o",
  "base_url": "https://api.openai.com/v1/",
  "model_name": "gpt-4o",
  "provider": "openai",
  "api_key": "sk-proj-..."
}
```

**Requirements:**
- OpenAI API key from https://platform.openai.com/api-keys
- `base_url` set to `https://api.openai.com/v1/`
- Valid `api_key` starting with `sk-`

**Available Models:**
- `gpt-4o` - Latest GPT-4 with vision
- `gpt-4o-mini` - Faster, cheaper variant
- `gpt-4-turbo` - Previous generation
- `gpt-4-vision-preview` - Vision preview

### 3. Anthropic Claude Models

```json
{
  "id": "claude-3-5-sonnet",
  "display_name": "Claude 3.5 Sonnet",
  "base_url": "",
  "model_name": "claude-3-5-sonnet-20241022",
  "provider": "anthropic",
  "api_key": "sk-ant-..."
}
```

**Requirements:**
- Anthropic API key from https://console.anthropic.com/settings/keys
- `base_url` must be empty string `""`
- Valid `api_key` starting with `sk-ant-`

**Available Models:**
- `claude-3-5-sonnet-20241022` - Most capable
- `claude-3-5-haiku-20241022` - Fast and efficient
- `claude-3-opus-20240229` - Previous flagship

### 4. Google Gemini Models

```json
{
  "id": "gemini-2-0-flash",
  "display_name": "Gemini 2.0 Flash",
  "base_url": "",
  "model_name": "gemini-2.0-flash-exp",
  "provider": "gemini",
  "api_key": "AIza..."
}
```

**Requirements:**
- Gemini API key from https://aistudio.google.com/app/apikey
- `base_url` must be empty string `""`
- Valid `api_key` starting with `AIza`

**Available Models:**
- `gemini-2.0-flash-exp` - Latest experimental
- `gemini-1.5-pro-latest` - Stable production
- `gemini-1.5-flash-latest` - Fast variant

---

## Adding New Models

### Example: Adding a Custom OpenAI-Compatible Model

```json
{
  "id": "my-custom-vlm",
  "display_name": "My Custom VLM Server",
  "base_url": "http://192.168.1.100:9000/v1/",
  "model_name": "custom/internvl-26b",
  "provider": "vllm",
  "api_key": "dummy"
}
```

### Example: Adding Azure OpenAI

```json
{
  "id": "azure-gpt-4o",
  "display_name": "Azure GPT-4o",
  "base_url": "https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT/",
  "model_name": "gpt-4o",
  "provider": "openai",
  "api_key": "YOUR_AZURE_API_KEY"
}
```

### Example: Adding Future Models

When new models are released:

```json
{
  "id": "gpt-5-preview",
  "display_name": "OpenAI GPT-5 Preview",
  "base_url": "https://api.openai.com/v1/",
  "model_name": "gpt-5-preview",
  "provider": "openai",
  "api_key": "sk-..."
}
```

**No code changes required!** Just add to `models.json` and reload.

---

## Security Best Practices

### 1. **Never Commit API Keys to Git**

```bash
# Add to .gitignore
echo "models.json" >> .gitignore
```

### 2. **Use Environment Variables (Alternative)**

While `models.json` supports hardcoded keys, you can also use environment variables:

```json
{
  "api_key": "${OPENAI_API_KEY}"
}
```

Then set environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

**Note:** Current implementation doesn't support `${VAR}` syntax yet. This is a future enhancement.

### 3. **Restrict File Permissions**

```bash
chmod 600 models.json  # Owner read/write only
```

### 4. **Use Separate Keys per Environment**

```
Development:   models.dev.json
Staging:       models.staging.json
Production:    models.prod.json
```

---

## Troubleshooting

### Error: "No models found in configuration"

**Cause:** `models.json` is malformed or empty

**Solution:**
1. Validate JSON syntax: https://jsonlint.com
2. Ensure `models` array exists and contains at least one model
3. Check for missing commas, brackets, or quotes

### Error: "API key not found"

**Cause:** API key is missing or invalid

**Solution:**
1. Check API key is correctly pasted (no extra spaces)
2. Verify API key is valid on provider's dashboard
3. For local models, use `"api_key": "dummy"`

### Error: "Model not loading after upload"

**Cause:** Configuration not reloaded

**Solution:**
1. Click "🔄 Reload Models" button
2. Or refresh the page (Ctrl+R / Cmd+R)
3. Or restart Docker container

### Models show default list instead of custom config

**Cause:** `models.json` not found or failed to load

**Solution:**
1. Check file is named exactly `models.json` (not `models.json.txt`)
2. Place file in project root directory
3. Check Docker logs for error messages:
   ```bash
   docker logs deepfake-detector-app
   ```

---

## Migration from Old System

### Before (Hardcoded in config.py)

```python
MODEL_CONFIGS = {
    "gpt-4o": {
        "api_key": os.getenv("OPENAI_API_KEY", "")
    }
}
```

### After (Dynamic JSON)

```json
{
  "models": [
    {
      "id": "gpt-4o",
      "api_key": "sk-YOUR_KEY_HERE"
    }
  ]
}
```

**Migration Steps:**
1. Create `models.json` from template
2. Copy API keys from environment variables or config.py
3. Update `base_url` values for local models
4. Upload and test

---

## Advanced Usage

### Multi-Environment Setup

**Directory Structure:**
```
project/
├── models.json               # Default (ignored by git)
├── models.json.example       # Template (committed)
├── configs/
│   ├── models.dev.json      # Development config
│   ├── models.prod.json     # Production config
│   └── models.test.json     # Testing config
```

**Load specific config:**
```python
from config import load_model_configs

# Load custom config
configs = load_model_configs("configs/models.prod.json")
```

### Model Aliases

Create multiple entries for same model with different settings:

```json
{
  "models": [
    {
      "id": "gpt-4o-fast",
      "display_name": "GPT-4o (Fast Mode)",
      "model_name": "gpt-4o",
      "provider": "openai",
      "api_key": "...",
      "_custom_params": {"temperature": 0.3}
    },
    {
      "id": "gpt-4o-precise",
      "display_name": "GPT-4o (Precise Mode)",
      "model_name": "gpt-4o",
      "provider": "openai",
      "api_key": "...",
      "_custom_params": {"temperature": 0.0}
    }
  ]
}
```

**Note:** `_custom_params` is not currently used but reserved for future enhancements.

---

## Example Configurations

### Minimal (Local Only)

```json
{
  "models": [
    {
      "id": "qwen3-local",
      "display_name": "Qwen3 VL 32B",
      "base_url": "http://localhost:8006/v1/",
      "model_name": "Qwen3-VL-32B-Instruct",
      "provider": "vllm",
      "api_key": "dummy"
    }
  ]
}
```

### Cloud-Only (No Local Models)

```json
{
  "models": [
    {
      "id": "gpt-4o",
      "display_name": "GPT-4o",
      "base_url": "https://api.openai.com/v1/",
      "model_name": "gpt-4o",
      "provider": "openai",
      "api_key": "sk-..."
    },
    {
      "id": "claude-3-5-sonnet",
      "display_name": "Claude 3.5 Sonnet",
      "base_url": "",
      "model_name": "claude-3-5-sonnet-20241022",
      "provider": "anthropic",
      "api_key": "sk-ant-..."
    }
  ]
}
```

### Hybrid (Local + Cloud)

```json
{
  "models": [
    {
      "id": "qwen3-local",
      "display_name": "Qwen3 VL (Local)",
      "base_url": "http://localhost:8006/v1/",
      "model_name": "Qwen3-VL-32B-Instruct",
      "provider": "vllm",
      "api_key": "dummy"
    },
    {
      "id": "gpt-4o-fallback",
      "display_name": "GPT-4o (Cloud Fallback)",
      "base_url": "https://api.openai.com/v1/",
      "model_name": "gpt-4o",
      "provider": "openai",
      "api_key": "sk-..."
    }
  ]
}
```

---

## Related Files

- [models.json.example](models.json.example) - Template configuration file
- [config.py](config.py#L7-L40) - Configuration loader implementation
- [app.py](app.py#L58-L107) - UI for uploading models.json
- [MODEL_CONFIG_GUIDE.md](MODEL_CONFIG_GUIDE.md) - This guide

---

## Future Enhancements

- [ ] Environment variable substitution (`${VAR}` syntax)
- [ ] Per-model custom parameters (temperature, top_p, etc.)
- [ ] Model validation and health checks
- [ ] Auto-discovery of local vLLM servers
- [ ] Encrypted API key storage
- [ ] Model performance benchmarking in UI

---

## Sign-Off

**Feature:** JSON-based model configuration
**Status:** ✅ Implementation complete
**Benefits:**
- No code changes needed to add models
- Centralized API key management
- Easy configuration sharing
- Future-proof for new models
