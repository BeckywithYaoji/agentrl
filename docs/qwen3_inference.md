# Qwen3 Inference Notes

## Environment

- Machine: Apple Silicon `arm64`
- Python: `3.14.6`
- MLX: installed, package does not expose a version attribute in this environment
- MLX-LM: `0.31.3`
- Candidate model: `mlx-community/Qwen3-0.6B-4bit`
- Resolved revision: `73e3e38d981303bc594367cd910ea6eb48349da8`
- Milestone 2B uses shared `load_qwen()` with this pinned revision. Chat generation
  uses `apply_chat_template(..., enable_thinking=False, add_generation_prompt=True)`
  and `stream_generate` with greedy sampling and 256-token limit. See
  `milestone2b_results.md` for protocol stopping, evidence injection and real results.
- `model.safetensors` size: `335450584` bytes
- `model.safetensors` SHA256: `392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2`

## API selected

The smoke script uses the verified MLX-LM public shape:

```python
from mlx_lm import generate, load
model, tokenizer = load(model_id)
text = generate(model, tokenizer, prompt=prompt, max_tokens=64, verbose=False)
```

The model-only smoke test was attempted with a plain text prompt. Chat-template behavior and Qwen3 thinking controls were not inferred or fabricated because model loading did not complete.

## Current limitation

The Hugging Face snapshot is now complete. The required `model.safetensors` is 335,450,584 bytes. The initial download attempt timed out, but the cache later completed and the model loaded successfully, including with `HF_HUB_OFFLINE=1`.
