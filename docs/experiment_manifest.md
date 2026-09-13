# Experiment Manifest

This manifest records the frozen experiment trail for the search-agent post-training project. The final mainline is the 0.6B pipeline; 1.7B work is archived as a scaling study.

| Milestone | Commit | Role | Data Used | Checkpoint / Output | Result |
|---|---|---|---|---|---|
| M6-B/C | `7688e53` | Formal 0.6B SFT | train/val only | `/root/autodl-tmp/checkpoints/agentrl_m6c/best` | val EM/F1 0.13 / 0.1616 |
| M6-D | `0ea4d28` | Formal 0.6B GRPO-R0 | train + val selection | `/root/autodl-tmp/checkpoints/agentrl_m6d/global_step_25`, HF export `/root/autodl-tmp/checkpoints/agentrl_m6d/best_hf` | val EM/F1 0.14 / 0.1716 |
| M6-E1 | `5189815` | R2 parser/reward alignment | train/val diagnostics | code fix only | `<information>` observation handling and `requires_search` propagation fixed |
| M6-E2 | `6f3a8c2` | Formal 0.6B GRPO-R2 | train + val selection | `/root/autodl-tmp/checkpoints/agentrl_m6e2/best` | val EM/F1 0.14 / 0.1671 |
| M6-F | `e442750` | Frozen Test/OOD/Counterfactual matrix | first frozen evaluation | `artifacts/milestone6f/frozen_eval/report.json` | Test Candidate EM: Base 0.33, SFT 0.50, R0 0.50, R2 0.51 |
| M6-G | `5e371d1` | Ablations and Milestone 6 report | frozen evaluation + declared ablations | `docs/milestone6_final_report.md` | no-search and concise-reminder ablations complete |
| M7-A | `63a930f` | 1.7B QLoRA/LoRA smoke | train-safe samples only | `/root/autodl-tmp/checkpoints/agentrl_m7a/` | load/save/reload/generation smoke passed |
| M7-B | `d4617ef` | Formal 1.7B LoRA SFT | train/val only | `/root/autodl-tmp/checkpoints/agentrl_m7b/best_adapter` | val EM/F1 0.15 / 0.1789 |
| M7-C | `1f08e06` | 1.7B GRPO feasibility smoke | train/val smoke | feasibility artifacts | identified memory pressure and feasibility constraints |
| M7-C2 | `4fb2480` | Memory-efficient LoRA-GRPO fix | train/val smoke | code fix and diagnostics | LoRA-only optimizer and memory-efficient path enabled |
| M7-C3 | `d40edb2` | Stability/checkpoint/resume smoke | train/val smoke | checkpoint/resume artifacts | resume-capable LoRA checkpoint path validated |
| M7-D | `8dc7461` | Formal 1.7B LoRA-GRPO-R0 | train + val selection | `/root/autodl-tmp/checkpoints/agentrl_m7d/best` | val EM/F1 0.11 / 0.1385; archived as negative scaling result |

## Frozen Data Discipline

- `data/sft/test.jsonl`, `data/sft/test_ood.jsonl`, and `data/sft/counterfactual.jsonl` were first unlocked at M6-F for final evaluation.
- Frozen results were not used to tune prompts, rewards, hyperparameters, or checkpoint selection after unsealing.
- M7 scaling work did not use Test/OOD/Counterfactual.

## Final Mainline

```text
Qwen3-0.6B -> HF/PyTorch SFT -> SearchXMLAgentLoop -> TinyBM25 -> GRPO-R0/R2 -> frozen evaluation -> ablations
```

The selected project conclusion is based on the 0.6B mainline. Qwen3-1.7B is retained as an appendix showing feasible LoRA scaling but non-improving short GRPO under the tested setup.
