# Logprob Troubleshooting Guide

**Issue:** Extreme negative logprobs (-15.625, -14.000) resulting in 0.0000 probabilities

---

## Understanding the Problem

### What You're Seeing

```
P(FAKE): -15.625 → 0.0000
P(REAL): -14.000 → 0.0000
```

### What This Means

1. **Logprobs are in log space** (natural logarithm)
2. **exp(-15.625) ≈ 1.5e-7** (extremely small, rounds to 0.0000 in display)
3. **exp(-14.000) ≈ 8.3e-7** (also extremely small)

These values indicate the REAL/FAKE tokens **are present** but have **very low probability** - meaning they're in the `top_logprobs` but not the primary output.

---

## Diagnostic Steps

### Step 1: Check Debug Tokens

After uploading an image, look for the debug message in the chat:

```
🔍 Debug: Top Tokens from Model

- `<token1>`: <logprob1>
- `<token2>`: <logprob2>
- `<token3>`: <logprob3>
...
```

**Key Questions:**
1. What is the first token in the list? (This is what the model actually output)
2. Are REAL/Real/real or FAKE/Fake/fake in the list?
3. What are their logprobs compared to the top token?

---

## Common Scenarios

### Scenario 1: Model Not Following Prompt

**Debug Output:**
```
- `The`: -0.5
- `I`: -2.1
- `This`: -3.4
- `REAL`: -15.625
- `FAKE`: -14.000
```

**Problem:** Model is trying to write a sentence instead of single-token output

**Cause:** System prompt not being followed correctly

**Solutions:**

**Option A: Strengthen System Prompt**

Edit `classifier.py`, line ~40:

```python
SYSTEM_PROMPT = """You are a forensic signal processing unit.

CRITICAL INSTRUCTION: Output EXACTLY ONE WORD. Nothing else.

You analyze three input images:
1. Original Photograph
2. ELA (Error Level Analysis) Map
3. FFT (Fast Fourier Transform) Spectrum

Analysis Rules:
- If FFT shows Grid/Starfield/Cross pattern → Output: FAKE
- If ELA shows uniform Rainbow static → Output: FAKE
- If Original shows physical inconsistencies → Output: FAKE
- If FFT is chaotic Starburst AND ELA is dark/edge-noise → Output: REAL

OUTPUT FORMAT: REAL or FAKE (ONE WORD ONLY)
"""
```

**Option B: Use Response Format (if API supports it)**

```python
response = self.client.chat.completions.create(
    model=self.model_name,
    messages=[...],
    temperature=0.0,
    max_tokens=1,
    logprobs=True,
    top_logprobs=5,
    response_format={"type": "text"}  # Some APIs support this
)
```

---

### Scenario 2: Token Casing Mismatch

**Debug Output:**
```
- `Real`: -0.3
- `Fake`: -0.7
- `REAL`: -15.625
- `FAKE`: -14.000
```

**Problem:** Model outputs `Real`/`Fake` (capitalized) but we're also checking `REAL`/`FAKE` (all caps)

**Cause:** Token list already includes these, but displaying wrong ones

**Solution:** Already handled! The code checks:
```python
REAL_TOKENS = ['REAL', ' REAL', 'Real', ' Real', 'real', ' real']
FAKE_TOKENS = ['FAKE', ' FAKE', 'Fake', ' Fake', 'fake', ' fake']
```

This should already work. The `-15.625` values are for tokens **NOT** matching.

---

### Scenario 3: Model Outputting Unexpected Token

**Debug Output:**
```
- `authentic`: -0.2
- `genuine`: -1.5
- `artificial`: -2.1
- `REAL`: -15.625
- `FAKE`: -14.000
```

**Problem:** Model using synonyms instead of exact REAL/FAKE

**Cause:** Model's training/vocabulary doesn't align with our tokens

**Solutions:**

**Option A: Add Synonyms to Token Lists**

Edit `classifier.py`:

```python
REAL_TOKENS = [
    'REAL', ' REAL', 'Real', ' Real', 'real', ' real',
    'Authentic', 'authentic', 'genuine', 'Genuine'
]
FAKE_TOKENS = [
    'FAKE', ' FAKE', 'Fake', ' Fake', 'fake', ' fake',
    'Artificial', 'artificial', 'Generated', 'generated'
]
```

**Option B: Map Token to Category**

```python
REAL_TOKEN_MAP = {
    'REAL': 'real', 'Real': 'real', 'real': 'real',
    'Authentic': 'real', 'authentic': 'real',
    'genuine': 'real', 'Genuine': 'real',
    'true': 'real', 'True': 'real'
}

FAKE_TOKEN_MAP = {
    'FAKE': 'fake', 'Fake': 'fake', 'fake': 'fake',
    'Artificial': 'fake', 'artificial': 'fake',
    'Generated': 'fake', 'generated': 'fake',
    'synthetic': 'fake', 'Synthetic': 'fake',
    'AI': 'fake', 'ai': 'fake'
}

# Then in parsing:
for logprob_obj in top_logprobs:
    token = logprob_obj.token
    logprob = logprob_obj.logprob

    if token in REAL_TOKEN_MAP and score_real is None:
        score_real = logprob
    elif token in FAKE_TOKEN_MAP and score_fake is None:
        score_fake = logprob
```

---

### Scenario 4: Whitespace Token Issues

**Debug Output:**
```
- ` Real`: -0.3    (note the leading space)
- ` Fake`: -0.8
- `Real`: -10.2    (no leading space)
- `REAL`: -15.625
```

**Problem:** Tokenizer includes/excludes leading whitespace inconsistently

**Cause:** Different tokenizers handle whitespace differently

**Solution:** Already handled! Our token lists include both:
```python
'REAL', ' REAL'  # With and without space
```

---

## Advanced Debugging

### Enable Verbose Logging

Edit `classifier.py`, add to `_parse_logprobs` method:

```python
# After extracting top_logprobs
print(f"\n=== LOGPROB DEBUG ===")
print(f"Model: {self.model_name}")
print(f"Token output: '{token_output}' (logprob: {logprobs_content.logprob:.3f})")
print(f"\nTop {len(all_tokens_debug)} tokens:")
for i, (token, logprob) in enumerate(all_tokens_debug, 1):
    print(f"  {i}. '{token}': {logprob:.3f} (prob: {math.exp(logprob):.6f})")
print(f"\nFound REAL: {score_real is not None} (logprob: {score_real if score_real else 'N/A'})")
print(f"Found FAKE: {score_fake is not None} (logprob: {score_fake if score_fake else 'N/A'})")
print(f"===================\n")
```

Then check Docker logs:
```bash
docker compose logs -f deepfake-detector
```

---

## Model-Specific Issues

### vLLM Models

Some vLLM setups may not properly support `logprobs=True` with vision models.

**Test:**
```bash
docker compose exec deepfake-detector python3 -c "
from openai import OpenAI
from config import MODEL_CONFIGS

# Test with your model
config = MODEL_CONFIGS['Qwen3-VL-32B-Instruct']
client = OpenAI(base_url=config['base_url'], api_key='dummy')

response = client.chat.completions.create(
    model=config['model_name'],
    messages=[{'role': 'user', 'content': 'Say REAL'}],
    max_tokens=1,
    logprobs=True,
    top_logprobs=5
)

print('Logprobs support:', hasattr(response.choices[0], 'logprobs'))
if hasattr(response.choices[0], 'logprobs'):
    print('Content:', response.choices[0].logprobs.content)
"
```

If this fails, vLLM might not support logprobs for your model.

---

## Workaround: Fallback to Text Parsing

If logprobs consistently fail, you can add a fallback:

Edit `classifier.py`, in `classify_image`:

```python
try:
    # Try logprob extraction
    result = classifier.classify_pil_image(...)

except Exception as e:
    # Fallback: parse text output
    response = self.client.chat.completions.create(
        model=self.model_name,
        messages=[...],
        temperature=0.0,
        max_tokens=10,  # Allow a few tokens
        logprobs=False  # Don't request logprobs
    )

    text_output = response.choices[0].message.content.strip().upper()

    # Simple text matching
    if 'FAKE' in text_output or 'ARTIFICIAL' in text_output or 'AI' in text_output:
        confidence = 0.9  # High confidence based on text
        is_ai = True
    elif 'REAL' in text_output or 'AUTHENTIC' in text_output:
        confidence = 0.1  # Low AI confidence
        is_ai = False
    else:
        confidence = 0.5  # Uncertain
        is_ai = False

    return {
        "is_ai": is_ai,
        "confidence_score": confidence,
        "classification": "AI-Generated" if is_ai else "Authentic",
        "token_output": text_output,
        "raw_logits": {"real": None, "fake": None},
        "raw_probs": {"real": None, "fake": None},
        "fallback_mode": True
    }
```

---

## Expected Debug Output (Working Correctly)

### Good Example 1: Clear FAKE Classification
```
🔍 Debug: Top Tokens from Model

- `FAKE`: -0.15
- `Fake`: -1.23
- `fake`: -2.45
- `REAL`: -5.67
- `Real`: -6.89

Token Output: `FAKE`
```

**Analysis:**
- Model clearly output `FAKE` (lowest logprob = highest probability)
- REAL alternatives have much lower probability
- Confidence calculation: exp(-0.15) / (exp(-0.15) + exp(-5.67)) ≈ 0.996 → 99.6% AI

### Good Example 2: Borderline Case
```
🔍 Debug: Top Tokens from Model

- `Real`: -0.69
- `Fake`: -0.73
- `REAL`: -1.12
- `FAKE`: -1.08
- `authentic`: -2.34

Token Output: `Real`
```

**Analysis:**
- Very close probabilities (exp(-0.69) ≈ 0.502, exp(-0.73) ≈ 0.482)
- Confidence: 0.482 / (0.502 + 0.482) ≈ 0.490 → 49% AI (borderline)
- This is exactly the "Timidity Bias" case - model is uncertain!

---

## Quick Fixes Summary

| Symptom | Quick Fix |
|---------|-----------|
| Model outputs sentences | Strengthen system prompt with "ONE WORD ONLY" |
| Synonyms instead of REAL/FAKE | Add synonyms to token lists |
| Whitespace issues | Already handled (includes ` REAL` variants) |
| vLLM doesn't support logprobs | Use fallback text parsing |
| Extreme negative values | Check if REAL/FAKE are actually in top_logprobs |

---

## Next Steps

1. **Rebuild Docker image** with updated code:
   ```bash
   docker compose down
   docker compose build
   docker compose up -d
   ```

2. **Upload test image** and check debug tokens

3. **Report findings:**
   - What are the top 5 tokens?
   - What is the actual token output?
   - Are REAL/FAKE in the list?

4. **Apply appropriate fix** based on debug output

---

## Contact & Support

If you continue having issues:
1. Capture the full debug token output
2. Note which model you're using
3. Share a screenshot of the chat with debug tokens
4. Check vLLM version: `docker compose exec deepfake-detector pip show vllm`

The debug token display will help us identify exactly why logprobs aren't working as expected!
