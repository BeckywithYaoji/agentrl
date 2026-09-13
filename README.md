# Search Agent Post-training

A search-augmented reasoning agent post-training project using HF SFT + veRL GRPO, with native multi-turn search rollout, verifiable rewards, frozen evaluation, and ablation analysis.

## Project Overview

This repository freezes the mainline result as the Qwen3-0.6B search agent. The project studies whether supervised fine-tuning and short GRPO post-training improve an XML-style search-and-answer policy for evidence-grounded QA.

The final mainline is:

```text
Qwen3-0.6B -> HF/PyTorch SFT -> SearchXMLAgentLoop -> query-sensitive BM25 retrieval -> veRL GRPO-R0/R2 -> Test/OOD/Counterfactual -> Ablation
```

The 1.7B work is archived as a scaling study appendix. It is not the final best model.

## Architecture

```text
Question -> Qwen3 -> <search> query -> BM25 -> <information> -> <answer>
```

Training chain:

```text
Base -> SFT -> GRPO-R0 / GRPO-R2
```

All formal agent runs use the native `SearchXMLAgentLoop` protocol with one XML action per turn. Search observations are returned inside `<information>...</information>` and are not supervised as assistant actions.

## Key Components

- `src/agentrl/retrieval.py`: TinyBM25 retrieval, including query-sensitive candidate retrieval.
- `src/agentrl/reward.py`: normalized answer metrics and R0/R2 reward helpers.
- `src/agentrl/search_agent.py`: SearchXML parsing, prompting, and multi-turn rollout helpers.
- `scripts/milestone6f_frozen_eval.py`: frozen Test/OOD/Counterfactual evaluation matrix.
- `scripts/milestone6g_ablations.py`: no-search and concise-reminder ablations.
- `scripts/milestone7d_lora_grpo_r0.py`: archived 1.7B LoRA-GRPO scaling run.

## Key Results

Main frozen Test result under Candidate BM25:

| Model | EM | Token F1 |
|---|---:|---:|
| Base | 0.33 | 0.4489 |
| SFT | 0.50 | 0.6072 |
| SFT+GRPO-R0 | 0.50 | 0.6115 |
| SFT+GRPO-R2 | 0.51 | 0.6148 |

Summary:

- Candidate-BM25 Test EM: `0.33 -> 0.50 -> 0.51`.
- Candidate-BM25 Test Token F1: `0.4489 -> 0.6072 -> 0.6148`.
- Search Decision Accuracy: `100%` after SFT on the formal 0.6B pipeline.
- OOD and Counterfactual Candidate BM25: SFT/R0/R2 reached `1.00 EM`.
- No-search OOD and Counterfactual: `0.00 EM`, confirming dependence on evidence retrieval.

## Scaling Appendix

Qwen3-1.7B LoRA/GRPO was evaluated after the 0.6B mainline was frozen. It is retained as a negative scaling result:

| Model | Val EM | Val Token F1 | Search Decision Acc |
|---|---:|---:|---:|
| 0.6B SFT | 0.13 | 0.1616 | 1.00 |
| 0.6B SFT+GRPO-R0 | 0.14 | 0.1716 | 1.00 |
| 1.7B LoRA SFT | 0.15 | 0.1789 | 0.99 |
| 1.7B LoRA-GRPO-R0 | 0.11 | 0.1385 | 0.95 |

The larger model showed a small SFT benefit, but 100-step sparse R0 LoRA-GRPO did not improve further and introduced search-policy drift.

## Documentation

- [Final project report](docs/final_project_report.md)
- [Experiment manifest](docs/experiment_manifest.md)
- [Checkpoint manifest](docs/checkpoint_manifest.md)
- [Resume summary](docs/resume_summary.md)
- [Interview notes](docs/interview_notes.md)
- [Milestone 6 final report](docs/milestone6_final_report.md)

## Limitations

- Final mainline uses Qwen3-0.6B.
- Experiments ran on a single RTX 4090.
- GRPO runs were limited to 100 optimizer steps.
- Train-only BM25 held-out evidence coverage is 0.
- Candidate BM25 is a controlled-corpus evaluation, not open-domain retrieval.
- R0/R2 differences are small.
- OOD and counterfactual splits are small.
