# Milestone 6 Final Experiment Report

## A. System Architecture

Qwen3-0.6B -> HF/PyTorch SFT -> veRL GRPO -> SearchXMLAgentLoop -> TinyBM25Retriever -> R0 / R2.

All agent evaluations use SearchXML actions (`<search>...</search>` and `<answer>...</answer>`) with deterministic generation unless an ablation explicitly states otherwise.

## B. Training Results

SFT ran 370 optimizer steps; train loss moved from 0.807655 to 0.187201. Best checkpoint was step 370 by validation teacher loss. Best val teacher loss was 0.243501; peak VRAM was 22248.0 MiB; wall time was 511.8s.
SFT locked-val EM/F1 was 0.13 / 0.1616.

GRPO-R0 reward curve by 25-step blocks: [0.49, 0.55, 0.57, 0.67]. Zero-variance/all-zero/all-perfect rates: 0.67 / 0.26 / 0.41. Peak VRAM: 19654 MiB; wall time: 690.3s.
GRPO-R2 reward curve by 25-step blocks: [0.48899999998509885, 0.54, 0.59, 0.6489999999850988]. Zero-variance/all-zero/all-perfect rates: 0.67 / 0.25 / 0.42. Peak VRAM: 19670 MiB; wall time: 698.1s.
R0 locked-val EM/F1: 0.14 / 0.1716; R2 locked-val EM/F1: 0.14 / 0.1671.

## C. Main Evaluation Table

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

Train-only BM25 has held-out support evidence coverage of 0 by construction, so its frozen-split failures are a retrieval-corpus coverage limitation as well as a policy test.
Candidate BM25 is a controlled-corpus evaluation: it measures behavior when evidence is present in each sample's candidate pool, not open-domain retrieval quality.

## D. Ablations

### No Search

| Model | Test EM/F1 | OOD EM/F1 | Counterfactual EM/F1 |
|---|---|---|---|
| Base | 0.00 / 0.0000 | 0.00 / 0.0000 | 0.00 / 0.0000 |
| SFT | 0.15 / 0.1500 | 0.00 / 0.0000 | 0.00 / 0.0000 |
| SFT+R0 | 0.15 / 0.1500 | 0.00 / 0.0000 | 0.00 / 0.0000 |
| SFT+R2 | 0.15 / 0.1500 | 0.00 / 0.0000 | 0.00 / 0.0000 |

No-search removes the retrieval step by setting search budget to 0. Candidate-BM25 gains disappear on OOD and counterfactual splits, showing those gains depend on search evidence rather than parametric recall.

### Concise Reminder ON vs OFF

| Mode | EM/F1 | Protocol | Exact Answer Format | R0 Zero Rate | Group Variance |
|---|---|---|---|---|---|
| on | 0.45 / 0.5310 | 1.00 | 1.00 | 0.55 | n/a deterministic |
| off | 0.40 / 0.5061 | 1.00 | 1.00 | 0.60 | n/a deterministic |

This ablation used SFT+R2 on the fixed first 20 validation examples with Candidate BM25. Only the concise-answer reminder was changed; generation and retrieval were unchanged.

### R0 vs R2

R2 is not clearly better than R0. On Test/Candidate BM25, R2 is +1pp EM over R0 and +0.0033 Token F1; OOD and Counterfactual Candidate BM25 are tied at 1.00 EM. Locked-val train-only EM is tied at 0.14, while R0 has slightly higher Token F1 (0.1716 vs 0.1671). Both keep protocol success, search precision, search recall, and search-decision accuracy at 1.00. R2 reward-hacking audit found 0 repeated-search groups, 0 high-search groups, 0 long-response groups, and 0 wrong-answer high-R2 rows; the measured R2 gain remains small.

### n=2 vs n=4

Omitted for compute efficiency. The main models remain the formal n=4 runs; no extra GRPO run was used to tune or reinterpret checkpoint selection.

## E. Key Findings

- SFT is the main performance source: Test/Candidate EM rises from Base 0.33 to SFT 0.50, with protocol/search decision accuracy improving to 1.00.
- GRPO after SFT brings only small extra gains while preserving stable search behavior: R0 keeps Test/Candidate EM at 0.50 and slightly improves F1; R2 reaches 0.51 EM.
- R2 is not materially superior to R0 in these runs; the observed Test/Candidate difference is about 1pp EM.
- After SFT, protocol and first search decision are already near saturated, limiting the extra value of R2 shaping.
- Many GRPO rollout groups have zero variance, reducing the proportion of useful policy-gradient signal.
- Retrieval corpus coverage is a major bottleneck: train-only BM25 has 0 held-out support evidence coverage, and OOD/counterfactual train-only EM stays at 0.
- When candidate evidence exists, OOD and counterfactual evidence use is strong for SFT/R0/R2, reaching 1.00 EM with Candidate BM25.

## F. Limitations

- Qwen3-0.6B only.
- Single RTX 4090.
- 100-step GRPO only.
- Train-only BM25 held-out evidence coverage is 0.
- Candidate retrieval is a controlled-corpus evaluation, not open-domain retrieval.
- R0/R2 differences are small.
- OOD and counterfactual splits are small.
