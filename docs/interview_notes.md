# Interview Notes

## Project Pitch

I built a compact search-augmented QA agent around Qwen3-0.6B. The model learns a simple XML protocol: emit `<search>query</search>`, receive BM25 evidence inside `<information>...</information>`, then emit `<answer>...</answer>`. The project covers supervised fine-tuning, veRL GRPO post-training, frozen evaluation, ablations, and a 1.7B scaling appendix.

## What Improved?

The main gain came from SFT. On frozen Test with Candidate BM25, EM improved from 0.33 for Base to 0.50 after SFT. GRPO after SFT was much smaller: R0 stayed at 0.50 EM with a small F1 increase, and R2 reached 0.51 EM.

## Why Was R2 Not a Clear Win?

R2 added reward components for answer correctness, protocol validity, first search decision correctness, and over-search penalty. But after SFT, protocol and search-decision behavior were already near saturated. Both R0 and R2 also had about 67% zero-variance rollout groups, so many GRPO groups produced little useful advantage signal.

## What Did Frozen Evaluation Show?

Train-only BM25 had 0 held-out support evidence coverage, so Test/OOD/Counterfactual failures under that protocol expose a retrieval-corpus coverage bottleneck. Candidate BM25 showed that when evidence is present in the candidate pool, the SFT/R0/R2 agents can use it well: OOD and Counterfactual EM reached 1.00 for those models.

## How Did You Avoid Retrieval Leakage?

Candidate BM25 was query-sensitive and ranked only with model-generated query plus document title/text. It did not use `is_support`, `correct_doc_position`, answer text, or gold labels for ranking. Support flags were used only after evaluation for hit@1/hit@3 diagnostics.

## Why Archive 1.7B as a Negative Result?

Qwen3-1.7B LoRA SFT was feasible and improved val EM from the 0.6B SFT's 0.13 to 0.15. However, the formal 100-step 1.7B LoRA-GRPO-R0 run dropped val EM to 0.11 and search decision accuracy from 0.99 to 0.95. That suggests the tested sparse-reward LoRA-GRPO setup was not a good scaling path yet.

## Lessons Learned

- A small, clear tool protocol can be stabilized effectively with SFT.
- Short GRPO needs enough reward variance; saturated or all-zero/all-perfect groups limit learning.
- Retrieval coverage can dominate model-side improvements in evidence-grounded QA.
- Controlled candidate retrieval is useful for diagnosing evidence use, but it is not open-domain retrieval.
- Scaling the model alone does not guarantee better post-training behavior under sparse rewards.

## Numbers To Remember

| Result | Value |
|---|---:|
| Test Candidate EM, Base | 0.33 |
| Test Candidate EM, SFT | 0.50 |
| Test Candidate EM, SFT+R2 | 0.51 |
| Test Candidate Token F1, Base | 0.4489 |
| Test Candidate Token F1, SFT+R2 | 0.6148 |
| OOD Candidate EM, SFT/R0/R2 | 1.00 |
| Counterfactual Candidate EM, SFT/R0/R2 | 1.00 |
| No-search OOD/Counterfactual EM | 0.00 |
| 0.6B R0/R2 zero-variance rate | about 0.67 |
| 1.7B SFT Val EM | 0.15 |
| 1.7B LoRA-GRPO-R0 Val EM | 0.11 |
