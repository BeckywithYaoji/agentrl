# Milestone 4 — Qwen3 Search Agent QLoRA SFT and Evaluation

## Status

Milestone 4 is complete. Stage 4A target-mask audit passed; the 32-example tiny run passed the overfit/reload smoke test; one full 4-bit Qwen3 LoRA run completed; and Base vs SFT replay evaluation used the same frozen protocol.

## Training

- Base: `mlx-community/Qwen3-0.6B-4bit`, revision `73e3e38d981303bc594367cd910ea6eb48349da8`
- Data: 1,480 train targets / 185 validation targets; no test, counterfactual, or OOD records in training.
- Objective: exact target-only causal loss; prompt and padding tokens receive zero loss.
- LoRA: last 16 layers, rank 8, scale 20, dropout 0; 2,883,584 trainable parameters / 596,049,920 total (0.484%).
- Run: seed 42, learning rate 1e-5, 1,600 micro-iterations, accumulation 4, one run only.
- Validation loss: 1.411 at iteration 0, 0.174 at the end. Peak memory: 2.52 GB.
- Adapter SHA256: `87b475ff05eedf5b52ff705052a86e0f21cc45d7c6892dbc4d226d74a5226944`.

## Base vs SFT

All generations used temperature 0, thinking disabled, max 256 new tokens, max 3 searches, identical prompts and replayed evidence.

| split | Base protocol / EM | SFT protocol / EM | SFT token F1 | SFT decision F1 |
|---|---:|---:|---:|---:|
| Milestone 2B (4) | 0.25 / 0.00 | 1.00 / 1.00 | 1.000 | n/a |
| test (100) | 0.20 / 0.02 | 1.00 / 0.42 | 0.576 | 0.919 |
| counterfactual (30) | 0.20 / 0.20 | 1.00 / 0.967 | 0.993 | 1.000 |
| OOD (30) | 0.767 / 0.40 | 1.00 / 0.967 | 0.993 | 1.000 |

The SFT model always emitted a valid first action and completed within the search-step budget. On the 100-case test split it learned the correct search-vs-direct boundary substantially better than Base, but retained a direct-answer false-positive rate of 1.0: all 15 direct-answer cases still searched.

## Failure analysis

The largest remaining error cluster is answer extraction under noisy or hard-negative evidence: test EM is 0.50 on noisy retrieval, 0.233 on single-search hard negatives, and 0.667 on direct-answer cases. Counterfactual and OOD behavior is strong under this replay protocol (29/30 exact match each), but this should not be interpreted as live-web robustness.

Raw per-case outputs and summaries are kept under `artifacts/milestone4_eval/`; training events and the adapter are under `artifacts/adapters/` and remain ignored build artifacts. No RL or additional SFT run was started.
