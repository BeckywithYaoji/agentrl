# Milestone 5A — Reward Design + GRPO Readiness Audit

## Status

PASS. No GRPO, PPO, RLVR update, GPU training, extra SFT, or sweep was run.

## Reward candidates

- R0: `answer_em` only; retained as the Search-R1-style outcome baseline.
- R1: `answer_em * protocol_valid`; format cannot earn reward without a correct answer.
- R2: correctness-dominant composite. Wrong answers receive `0` (or `-0.1` for severe protocol invalidity); correct answers start at `1.0`, with only small protocol, decision, and excess-search deductions, clipped to `[-0.1, 1.0]`.

The adversarial suite passes: wrong perfect-format answers never receive positive reward, direct correct answers beat search-everything, redundant searches are lower than efficient search, aliases work, duplicate answer tags are invalid, and substring stuffing does not receive exact match.

## Offline audit

The saved Base/SFT Milestone 4 trajectories were audited without changing them. On the real frozen Test 100, SFT R2 mean was `0.408`; correct trajectories averaged `0.971` while incorrect trajectories averaged `0.000`. Base R2 mean was `-0.084`; its correct trajectories averaged `0.680` and incorrect trajectories `-0.100`. Counterfactual/OOD are synthetic control audits, not real-world generalization estimates.

## Validation stochastic probe

The SFT policy was sampled on 24 validation prompts × 4 rollouts with temperature `0.7`, top-p `0.95`, max 256 tokens, and max 3 search steps. ReplayRetriever was used only to study reward distribution and protocol/answer variation; this does not test learnable query quality.

| Reward | Mean | Std | Zero-variance groups | All-zero | All-perfect |
|---|---:|---:|---:|---:|---:|
| R0 | 0.365 | 0.481 | 70.8% | 50.0% | 20.8% |
| R1 | 0.365 | 0.481 | 70.8% | 50.0% | 20.8% |
| R2 | 0.358 | 0.474 | 66.7% | 50.0% | 16.7% |

Overall protocol success was `1.000`, answer EM `0.365`, search-decision accuracy `0.844`, and average search count `0.906`. The probe shows usable but sparse group signal: half of groups are all-zero, while R2 slightly reduces zero-variance and all-perfect collapse without inventing format reward.

## Recommendation

Use R0 as the recommended RL baseline and R2 as the reward-design ablation. Keep R1 as a protocol-gating diagnostic. Do not select a final reward from frozen test/OOD/counterfactual results.

The future training environment must be query-sensitive: `search(query) -> documents`, with garbage queries unable to reliably return gold evidence, no ground-truth leakage, and no reward based on document position. ReplayRetriever is explicitly evaluation/probe-only. The MLX adapter is not assumed compatible with PyTorch/Hugging Face veRL; the safest GPU route is to reproduce SFT in HF format before GRPO.

Artifacts: `artifacts/milestone5a/reward_audit.json`, `artifacts/milestone5a/grpo_probe.jsonl`, `artifacts/milestone5a/grpo_probe_summary.json`, and `artifacts/milestone5a/rl_schema_preview.jsonl`.
