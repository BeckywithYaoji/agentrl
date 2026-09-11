# Qwen3 Inference Notes

## Environment

- Machine: Apple Silicon `arm64`
- Python: `3.14.6`
- MLX: installed, package does not expose a version attribute in this environment
- MLX-LM: `0.31.3`
- Candidate model: `mlx-community/Qwen3-0.6B-4bit`

## API selected

The smoke script uses the verified MLX-LM public shape:

```python
from mlx_lm import generate, load
model, tokenizer = load(model_id)
text = generate(model, tokenizer, prompt=prompt, max_tokens=64, verbose=False)
```

The model-only smoke test was attempted with a plain text prompt. Chat-template behavior and Qwen3 thinking controls were not inferred or fabricated because model loading did not complete.

## Current limitation

The Hugging Face snapshot metadata and small files downloaded successfully, but the required `model.safetensors` is 335,450,584 bytes and was not received by either MLX-LM resume download or a direct resumable request within the available network window. Consequently no Qwen3 generation, chat-template inspection, or Search-Agent integration was performed.
