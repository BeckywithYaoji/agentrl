# Final Project Report

## Executive Summary

This project built and evaluated a compact search-augmented QA agent with explicit XML tool use. The frozen mainline is Qwen3-0.6B trained with HF/PyTorch SFT, followed by short veRL GRPO runs using native multi-turn `SearchXMLAgentLoop` rollouts and TinyBM25 retrieval.

The strongest effect came from SFT. On frozen Test with Candidate BM25, EM improved from 0.33 for the base model to 0.50 after SFT. GRPO added only small gains after SFT: R0 kept EM at 0.50 with slightly higher Token F1, and R2 reached 0.51 EM. R2 was not clearly better than R0.

## System Architecture

```text
Qwen3-0.6B
-> HF/PyTorch SFT
-> veRL GRPO
-> SearchXMLAgentLoop
-> TinyBM25Retriever
-> R0 / R2 rewards
```

Runtime agent flow:

```text
Question -> model emits <search>query</search> -> BM25 returns <information> -> model emits <answer>answer</answer>
```

The parser treats `<information>` as an observation, not as an assistant action. SFT masking supervises assistant `<search>` and `<answer>` tokens while masking system/user messages, observations, and padding.

## Training Results

| Stage | Start | Steps | Selection | Main Result |
|---|---|---:|---|---|
| 0.6B SFT | Qwen/Qwen3-0.6B | 370 | best val teacher loss | locked-val EM/F1 0.13 / 0.1616 |
| 0.6B GRPO-R0 | M6-C SFT best | 100 | val-only rule | locked-val EM/F1 0.14 / 0.1716 |
| 0.6B GRPO-R2 | M6-C SFT best | 100 | val-only rule | locked-val EM/F1 0.14 / 0.1671 |

SFT training loss moved from 0.807655 to 0.187201; best val teacher loss was 0.243501. SFT peak VRAM was 22,248 MiB and wall time was 511.8s.

GRPO-R0 reward curve by 25-step blocks was `[0.49, 0.55, 0.57, 0.67]`. Zero-variance/all-zero/all-perfect group rates were `0.67 / 0.26 / 0.41`. Peak VRAM was 19,654 MiB and wall time was 690.3s.

GRPO-R2 reward curve by 25-step blocks was `[0.489, 0.540, 0.590, 0.649]`. Zero-variance/all-zero/all-perfect group rates were `0.67 / 0.25 / 0.42`. Peak VRAM was 19,670 MiB and wall time was 698.1s.

## Main Frozen Evaluation

All four models used the same SearchXML prompt, deterministic generation, normalization, max search steps, top-k, and metric code. Test/OOD/Counterfactual were frozen and were not used for training.

### Test

| Model | Train-only EM/F1 | Candidate EM/F1 | Candidate hit@1/@3 | Protocol | Search Acc |
|---|---|---|---|---|---|
| Base | 0.00 / 0.0700 | 0.33 / 0.4489 | 0.6600 / 0.8000 | 0.98 / 0.94 | 0.85 / 0.85 |
| SFT | 0.15 / 0.1751 | 0.50 / 0.6072 | 0.8706 / 0.9412 | 1.00 / 1.00 | 1.00 / 1.00 |
| SFT+R0 | 0.15 / 0.1794 | 0.50 / 0.6115 | 0.8706 / 0.9412 | 1.00 / 1.00 | 1.00 / 1.00 |
| SFT+R2 | 0.15 / 0.1784 | 0.51 / 0.6148 | 0.8588 / 0.9412 | 1.00 / 1.00 | 1.00 / 1.00 |

### OOD

| Model | Train-only EM/F1 | Candidate EM/F1 | Candidate hit@1/@3 | Protocol | Search Acc |
|---|---|---|---|---|---|
| Base | 0.00 / 0.0000 | 0.00 / 0.1932 | 1.0000 / 1.0000 | 1.00 / 1.00 | 1.00 / 1.00 |
| SFT | 0.00 / 0.0000 | 1.00 / 1.0000 | 1.0000 / 1.0000 | 1.00 / 1.00 | 1.00 / 1.00 |
| SFT+R0 | 0.00 / 0.0000 | 1.00 / 1.0000 | 1.0000 / 1.0000 | 1.00 / 1.00 | 1.00 / 1.00 |
| SFT+R2 | 0.00 / 0.0000 | 1.00 / 1.0000 | 1.0000 / 1.0000 | 1.00 / 1.00 | 1.00 / 1.00 |

### Counterfactual

| Model | Train-only EM/F1 | Candidate EM/F1 | Candidate hit@1/@3 | Protocol | Search Acc |
|---|---|---|---|---|---|
| Base | 0.00 / 0.0913 | 1.00 / 1.0000 | 1.0000 / 1.0000 | 0.97 / 1.00 | 1.00 / 1.00 |
| SFT | 0.00 / 0.0000 | 1.00 / 1.0000 | 1.0000 / 1.0000 | 1.00 / 1.00 | 1.00 / 1.00 |
| SFT+R0 | 0.00 / 0.0000 | 1.00 / 1.0000 | 1.0000 / 1.0000 | 1.00 / 1.00 | 1.00 / 1.00 |
| SFT+R2 | 0.00 / 0.0000 | 1.00 / 1.0000 | 1.0000 / 1.0000 | 1.00 / 1.00 | 1.00 / 1.00 |

Train-only BM25 has held-out support evidence coverage of 0 by construction. Candidate BM25 is a controlled-corpus protocol where each sample's metadata documents form the retrieval pool; ranking uses the generated query and document text/title only, not labels or support flags.

## Ablations

### No Search

| Model | Test EM/F1 | OOD EM/F1 | Counterfactual EM/F1 |
|---|---|---|---|
| Base | 0.00 / 0.0000 | 0.00 / 0.0000 | 0.00 / 0.0000 |
| SFT | 0.15 / 0.1500 | 0.00 / 0.0000 | 0.00 / 0.0000 |
| SFT+R0 | 0.15 / 0.1500 | 0.00 / 0.0000 | 0.00 / 0.0000 |
| SFT+R2 | 0.15 / 0.1500 | 0.00 / 0.0000 | 0.00 / 0.0000 |

No-search removed the retrieval step. Candidate-BM25 gains disappeared on OOD and Counterfactual, indicating that those results depend on evidence use rather than parametric recall.

### Concise Reminder ON/OFF

| Mode | EM/F1 | Protocol | Exact Answer Format | R0 Zero Rate | Group Variance |
|---|---|---:|---:|---:|---|
| ON | 0.45 / 0.5310 | 1.00 | 1.00 | 0.55 | n/a deterministic |
| OFF | 0.40 / 0.5061 | 1.00 | 1.00 | 0.60 | n/a deterministic |

This used SFT+R2 on the fixed first 20 validation examples with Candidate BM25. Only the concise-answer reminder changed.

### R0 vs R2

R2 was not materially superior to R0. Test/Candidate EM differed by about 1 percentage point, OOD and Counterfactual Candidate BM25 were tied at 1.00 EM, and locked-val EM was tied at 0.14. Both rewards had about 67% zero-variance rollout groups, limiting useful GRPO advantage signal.

## Scaling Study Appendix: Qwen3-1.7B

The 1.7B branch was run after the 0.6B mainline was completed and is archived as a scaling feasibility study. It did not replace the 0.6B mainline.

| Model | Val EM | Val Token F1 | Protocol | Search Decision Acc |
|---|---:|---:|---:|---:|
| 0.6B SFT | 0.13 | 0.1616 | 1.00 | 1.00 |
| 0.6B SFT+GRPO-R0 | 0.14 | 0.1716 | 1.00 | 1.00 |
| 1.7B LoRA SFT | 0.15 | 0.1789 | 1.00 | 0.99 |
| 1.7B LoRA-GRPO-R0 | 0.11 | 0.1385 | 1.00 | 0.95 |

1.7B LoRA SFT used 3,211,264 trainable parameters, a 0.1863% trainable ratio, and peaked at 19,830 MiB. The formal 1.7B LoRA-GRPO-R0 run used 100 optimizer steps from the 1.7B SFT adapter, saved adapter-only best/latest checkpoints of about 60.30 MiB each, and selected step 25 by the locked val-only rule. Reward mean/std was 0.5425 / 0.4169 with 0.65 zero-variance, 0.29 all-zero, and 0.36 all-perfect group rates.

Conclusion: the larger model capacity gave a small SFT benefit, but sparse R0 plus 100-step LoRA-GRPO did not improve further and showed search-policy drift. The final model claim remains the 0.6B mainline.

## Key Findings

- SFT is the main source of performance improvement.
- GRPO after SFT provides only small additional gains in this setup.
- R2 is not clearly better than R0.
- After SFT, protocol success and search-decision accuracy are near saturated, reducing the headroom for R2 shaping.
- Many rollout groups have zero variance, limiting the fraction of useful policy-gradient updates.
- Retrieval coverage is a major bottleneck: train-only BM25 has 0 held-out support evidence coverage.
- When candidate evidence is present, SFT/R0/R2 use it strongly on OOD and Counterfactual.
- 1.7B LoRA scaling was feasible, but the short LoRA-GRPO run was not beneficial under these constraints.

## Limitations

- Qwen3-0.6B is the final mainline model.
- Experiments used a single RTX 4090.
- GRPO was limited to 100 optimizer steps.
- Train-only BM25 held-out evidence coverage is 0.
- Candidate retrieval is controlled-corpus evaluation.
- R0/R2 differences are small.
- OOD and Counterfactual splits are small.
- 1.7B results use LoRA adapters and should be interpreted as a scaling feasibility study, not a tuned final model.
