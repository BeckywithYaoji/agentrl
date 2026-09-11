# Search-R1 Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve and inspect upstream Search-R1, document its real search architecture, and run one deterministic local search interaction on Mac.

**Architecture:** Keep upstream code under `third_party/search-r1/` and add only source-backed documentation plus a thin deterministic demo. Prefer upstream utilities; if their runtime requires unavailable services, use a small adapter that preserves the observed protocol and state the boundary explicitly.

**Tech Stack:** Git, Python 3, upstream Search-R1 source/configuration, standard library where possible.

**Spec:** `docs/superpowers/specs/2026-09-11-search-r1-milestone-1-design.md`

## Global Constraints

- Keep upstream Search-R1 unchanged.
- Avoid CUDA, NVIDIA-only packages, veRL execution, model downloads, and distributed training.
- Make each change runnable and verifiable before the next change.
- Commit after each verified milestone-sized increment.

### Task 1: Import and inventory upstream Search-R1

**Files:**
- Create: `third_party/search-r1/` via a git clone
- Create: `docs/search_r1_source_inventory.md`

- [ ] Clone the official Search-R1 repository into `third_party/search-r1/` without editing it.
- [ ] Record the upstream commit and inspect its top-level files, dependency manifests, and relevant search/data/RL paths.
- [ ] Write a short source inventory naming exact files for dataset preprocessing, prompts, search environment, retriever, rollout, reward, and RL trainer.
- [ ] Run `git -C third_party/search-r1 status --short` and confirm the upstream checkout is clean.
- [ ] Commit with `git add third_party/search-r1 docs/search_r1_source_inventory.md && git commit -m "chore: add upstream Search-R1 reference"`.

### Task 2: Document the architecture from source

**Files:**
- Create: `docs/search_r1_architecture.md`

- [ ] Read the exact source files identified in Task 1 and trace question, ground truth, prompt, search action, observation, multi-turn continuation, and final answer formats.
- [ ] Trace retriever model, index, corpus, and server configuration, distinguishing local-compatible pieces from service/GPU-dependent pieces.
- [ ] Locate veRL, rollout, reward, PPO/GRPO entry points without running training.
- [ ] Fill the required data-flow diagram, module table, search protocol section, and Mac compatibility table with file-backed facts.
- [ ] Run a placeholder scan with `rg -n "TBD|TODO|guess|probably" docs/search_r1_architecture.md` and remove all matches.
- [ ] Commit with `git add docs/search_r1_architecture.md && git commit -m "docs: map Search-R1 architecture"`.

### Task 3: Implement and verify the minimal interaction

**Files:**
- Create: `scripts/minimal_search_demo.py`
- Create: `tests/test_minimal_search_demo.py` when a focused test is practical

- [ ] Define a small deterministic interface for one question, one search action, one observation, and one final answer; keep it limited to the observed Search-R1 protocol.
- [ ] Reuse an upstream runnable component if it works without external services; otherwise provide a clearly labeled local stub/adaptor using a fixed document so the interaction remains deterministic.
- [ ] Add a test or command assertion covering the complete flow and the visible observation.
- [ ] Run the focused test and then `python scripts/minimal_search_demo.py` on the Mac environment.
- [ ] Commit with `git add scripts tests && git commit -m "feat: add minimal search interaction demo"`.

### Task 4: Final verification

**Files:**
- Modify: only files required by verification findings

- [ ] Run `git diff --check`.
- [ ] Verify `git -C third_party/search-r1 status --short` remains clean.
- [ ] Re-run the demo and focused tests from a clean project state.
- [ ] Confirm no Qwen3, CUDA, veRL training, SimpleTIR, or large model download was introduced.
- [ ] Report commit hashes, run commands, and any explicitly documented upstream limitation.
