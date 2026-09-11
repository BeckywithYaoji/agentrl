# Search-R1 Milestone 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Milestone 1 hard-coded action with a real local Qwen3/MLX inference loop while preserving deterministic tests.

**Architecture:** Keep the current parser and mock environment, add one thin MLX-LM adapter, and place the integration loop in one script. Model weights and caches remain outside Git; tests use a fake generator.

**Tech Stack:** Python 3, MLX, MLX-LM, standard library unittest, Qwen3-0.6B.

**Spec:** `docs/superpowers/specs/2026-09-11-search-r1-milestone-2-design.md`

## Global Constraints

- No SFT, LoRA, GRPO, PPO, RLVR, veRL training, vLLM, CUDA, web APIs, or large corpus.
- Prefer Qwen3-0.6B; only consider 1.7B after documenting a 0.6B failure.
- Never hard-code a model search action in the integration loop.
- Do not commit model weights, caches, `.DS_Store`, or Python caches.

### Task 1: Environment and provenance

**Files:** `docs/search_r1_source_inventory.md`, `.gitignore`, local `.venv/`

- [ ] Record Python version and machine architecture.
- [ ] Create `.venv` if absent and install only `mlx` and `mlx-lm`.
- [ ] Record installed versions and model identifier resolution.
- [ ] Add caches, artifacts, `.venv`, and Python caches to `.gitignore`.
- [ ] Record the Search-R1 source acquisition details and a reproducible current-snapshot hash; use `UNKNOWN` for unavailable original tarball hash or commit.
- [ ] Run the environment import checks and commit.

### Task 2: Model smoke test and inference notes

**Files:** `scripts/qwen_smoke_test.py`, `docs/qwen3_inference.md`

- [ ] Implement a small MLX-LM call for `Question: What is 2 + 2?` with model loading and latency measurement.
- [ ] Run it and stop if model loading or generation fails.
- [ ] Document the actual API, chat template behavior, thinking setting, generation parameters, and Mac limitations.
- [ ] Commit only after the real smoke test succeeds.

### Task 3: Adapter and deterministic loop tests

**Files:** `src/agentrl/qwen_agent.py`, `scripts/qwen_search_demo.py`, `tests/test_qwen_search_demo.py`

- [ ] Add the thin adapter around the verified MLX-LM API.
- [ ] Implement the loop with max search steps, existing parser, mock retrieval, observation append, answer termination, invalid-output stop, JSONL trajectory logging, and metrics.
- [ ] Add FakeAgent tests for all required loop behaviors and counterfactual evidence use.
- [ ] Run standard-library unit tests and commit.

### Task 4: Real trajectory and report

**Files:** `docs/milestone2_results.md`, `artifacts/milestone2/` (ignored runtime output)

- [ ] Run the real demo on three mock questions and the counterfactual question.
- [ ] Run a no-search baseline separately and record it without modifying the agent loop.
- [ ] Record zero-shot/few-shot setting, trajectories, metrics, failures, and acceptance status.
- [ ] Verify clean Git state and commit only source/docs, not runtime artifacts or model files.
