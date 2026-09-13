# Checkpoint Manifest

This file records which checkpoints matter after project closeout. Large checkpoints live outside Git under `/root/autodl-tmp/checkpoints/`.

| Path | Type | Role | Keep? | Notes |
|---|---|---|---|---|
| `/root/autodl-tmp/checkpoints/agentrl_m6c/best` | full HF model | 0.6B SFT best | Keep | Start checkpoint for formal 0.6B GRPO; final comparison baseline |
| `/root/autodl-tmp/checkpoints/agentrl_m6d/global_step_25` | veRL checkpoint | 0.6B GRPO-R0 selected checkpoint | Keep | Selected by val-only rule |
| `/root/autodl-tmp/checkpoints/agentrl_m6d/best_hf` | HF export | 0.6B GRPO-R0 evaluator checkpoint | Keep | Used by frozen evaluation |
| `/root/autodl-tmp/checkpoints/agentrl_m6e2/best` | veRL checkpoint | 0.6B GRPO-R2 selected checkpoint | Keep | Selected by val-only rule |
| `/root/autodl-tmp/checkpoints/agentrl_m6f_tmp/r2_hf` | HF export | 0.6B GRPO-R2 evaluator checkpoint | Optional keep | Recreated from M6-E2 best if needed |
| `/root/autodl-tmp/checkpoints/agentrl_m7a/` | LoRA smoke artifacts | 1.7B feasibility smoke | Optional archive | Smoke-only proof of load/save/reload/generation |
| `/root/autodl-tmp/checkpoints/agentrl_m7b/best_adapter` | LoRA adapter | 1.7B LoRA SFT | Archive | Scaling appendix; val EM/F1 0.15 / 0.1789 |
| `/root/autodl-tmp/checkpoints/agentrl_m7d/best` | LoRA/veRL adapter checkpoint | 1.7B LoRA-GRPO-R0 selected checkpoint | Archive | Negative scaling result; size 60.30 MiB |
| `/root/autodl-tmp/checkpoints/agentrl_m7d/latest` | LoRA/veRL adapter checkpoint | 1.7B LoRA-GRPO-R0 latest | Optional archive | Resume checkpoint; size 60.30 MiB |

## Git Tracking Policy

- Checkpoints, Hugging Face cache, venvs, and normal run artifacts should not be tracked in Git.
- `.gitignore` excludes `.venv/`, `.hf-cache/`, `artifacts/`, `__pycache__/`, `*.py[cod]`, and `.DS_Store`.
- A few early `artifacts/milestone2b/*` files are already tracked historical fixtures; no large checkpoint/cache files are tracked at closeout.

## Reproducibility Anchors

- Final source closeout commit: recorded by the release commit and tag `v1.0.0`.
- Main frozen report: `docs/final_project_report.md`.
- Detailed Milestone 6 report: `docs/milestone6_final_report.md`.
- Frozen evaluation aggregate: `artifacts/milestone6f/frozen_eval/report.json`.
- Milestone 7 scaling comparison: `artifacts/milestone7d/scaling_val_comparison.json`.
