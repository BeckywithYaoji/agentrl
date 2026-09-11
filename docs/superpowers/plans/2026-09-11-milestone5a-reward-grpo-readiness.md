# Milestone 5A Reward and GRPO Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement framework-independent R0/R1/R2 rewards, adversarial tests, offline audits, a validation-only stochastic GRPO probe, and compatibility documentation without starting any RL or SFT training.

**Architecture:** `src/agentrl/reward.py` owns parsing, normalized EM/alias matching, F1 diagnostics, decision and search-cost metrics, and the three reward candidates. Scripts adapt frozen Milestone 4 JSONL trajectories and validation samples into audit/probe artifacts; the veRL adapter remains a thin dependency-free wrapper. Documentation freezes evaluation policy and specifies a query-sensitive future training environment.

**Tech Stack:** Python standard library, existing agentrl normalization/loop utilities, MLX inference only for validation rollout generation, JSONL artifacts, vendored Search-R1 source inspection.

**Spec:** `/Users/yaoji/.codex/attachments/b1c68783-0d5d-4e24-bb58-de4761a3d69b/pasted-text.txt`

## Global Constraints

- Do not run GRPO, PPO, RLVR updates, veRL training, GPU training, extra SFT, or hyperparameter sweeps.
- Frozen Test 100, Counterfactual 30, OOD 30, and Milestone 2B are audit-only.
- The 24-prompt stochastic probe uses validation only, SFT policy, K=4, temperature 0.7, top_p 0.95 where supported.
- ReplayRetriever is evaluation/probe-only and must not be presented as a final RL environment.

### Task 1: Reward core and adversarial unit tests

Create `src/agentrl/reward.py` with `RewardInput`, `parse_trajectory`, `normalized_em`, `token_f1`, `compute_reward`, and `reward_summary`; implement R0=`answer_em`, R1=`answer_em * protocol_valid`, and correctness-dominant R2 with the specified constants and clipping. Add `tests/test_reward_hacking.py` covering all cases A-I, reward ordering, aliases, malformed protocol, decision correctness, and search cost.

### Task 2: Offline audit and frozen-policy documentation

Create `scripts/analyze_rewards.py` to audit existing Base/SFT JSONL files without mutation, and write `docs/evaluation_policy.md`, `docs/rl_environment.md`, and `docs/reward_hacking.md`. Include frozen dataset hashes, replay exploit, query-sensitive contract, retriever recommendation, and MLX-to-veRL checkpoint boundary.

### Task 3: veRL adapter and schema preview

Create `src/agentrl/verl_reward_adapter.py` as a dependency-free thin wrapper around the reward core, with a unit test importing it on CPU. Create a 10-row validation-only `artifacts/milestone5a/rl_schema_preview.jsonl` containing `data_source`, `prompt`, `ground_truth`, `extra_info`, aliases, sample ID, and search label.

### Task 4: Validation stochastic rollout probe

Create `scripts/grpo_rollout_probe.py` using the SFT adapter, validation-only stratified sampling up to 24 prompts, replay evidence, stochastic generation, and raw rollout persistence. Compute per-group R0/R1/R2 distributions, zero-variance/all-zero/all-perfect rates, protocol/EM/decision/search metrics, and save `artifacts/milestone5a/grpo_probe_summary.json`.

### Task 5: Verification and final report

Run all tests, run the reward audit and probe, verify no frozen-set IDs enter the probe, inspect reward separation and ordering, and write `docs/milestone5a_results.md`. Record that the probe is a readiness smoke test, not a generalization claim. Commit only source/tests/scripts/docs and the small schema preview; keep model/cache artifacts ignored.
