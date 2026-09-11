# Milestone 2A Report

## 1. Git

- base: `d0b0a89`
- final: pending this commit
- branch: `master`
- working tree: clean before documentation update

## 2. Environment

- Python: `3.14.6`
- MLX: installed
- MLX-LM: `0.31.3`
- huggingface_hub: `1.31.0`
- hf_xet: `1.6.0`
- architecture: `arm64`

## 3. Download

- repo: `mlx-community/Qwen3-0.6B-4bit`
- revision: `73e3e38d981303bc594367cd910ea6eb48349da8`
- cache: Hugging Face default cache under `~/.cache/huggingface/hub`
- model.safetensors: `335450584` bytes
- SHA256: `392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2`
- download method: `huggingface_hub` cache/resume, invoked through MLX-LM
- `HF_HUB_DOWNLOAD_TIMEOUT`: `120`
- `HF_HUB_ETAG_TIMEOUT`: `30`
- Xet: installed and available; no Xet-disabled fallback was needed

## 4. First Smoke Test

Command:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30 \
  .venv/bin/python scripts/qwen_smoke_test.py
```

Result: PASS. Model loading succeeded in 1.13 seconds and generation succeeded in 0.62 seconds.

Output:

```text
Let's see, 2 + 2 is 4. So the answer is 4. ...
```

## 5. Cached Smoke Test

Command:

```bash
.venv/bin/python scripts/qwen_smoke_test.py
```

Result: PASS. The loader completed without downloading model bytes; the cache was reused. Loading took 1.14 seconds and generation took 0.52 seconds.

## 6. Offline Smoke Test

Command:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/qwen_smoke_test.py
```

Result: PASS. Loading took 0.55 seconds and generation took 0.55 seconds.

## 7. Unit Tests

```text
python3 -m unittest discover -s tests -v
Ran 2 tests ... OK
```

## 8. Files Modified

- `docs/qwen3_inference.md`
- `docs/milestone2a_results.md`

No model files, Hugging Face cache, or weights were added to Git.

## 9. Problems

The initial large-file transfer timed out. The existing cache subsequently completed; no code workaround or unofficial mirror was used.

## 10. Milestone Result

PASS. Model acquisition, cached reload, offline reload, generation, and regression tests all passed. Agent integration was not started.
