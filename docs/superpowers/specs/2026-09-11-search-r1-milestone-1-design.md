# Search-R1 Milestone 1 Design

## Goal

Understand the upstream Search-R1 search interaction path and prove one minimal question → search action → observation → answer flow on the local Mac without loading a large model or running training.

## Scope

This milestone includes only:

- preserving upstream Search-R1 under `third_party/search-r1/`;
- reading and documenting its dataset, prompt, search protocol, search environment, retriever, rollout, reward, and RL entry points;
- determining which dependencies are usable on Apple Silicon;
- a deterministic demo using a hard-coded search action.

It excludes Qwen3 inference, SFT, GRPO/RLVR execution, veRL changes, SimpleTIR, CUDA setup, and large model or corpus downloads.

## Architecture

The repository keeps upstream code isolated and adds only project-owned documentation and a thin demo. The demo should reuse an upstream environment or retriever interface when that can run locally; otherwise it uses the smallest faithful adapter around the upstream search protocol and clearly records the limitation. No new general-purpose agent abstraction is introduced.

Expected flow, to be corrected from source inspection:

```text
Question
  → hard-coded agent search action
  → Search-R1-compatible search environment / retriever
  → observation
  → deterministic answer display
```

## Deliverables

- `docs/search_r1_architecture.md`: source-backed architecture map and Mac compatibility table.
- `scripts/minimal_search_demo.py`: one runnable deterministic interaction.
- focused tests or a command-level verification for the demo, depending on the upstream dependency shape.

## Constraints

- Keep upstream Search-R1 unchanged.
- Avoid CUDA, NVIDIA-only packages, veRL execution, model downloads, and distributed training.
- Make each change runnable and verifiable before the next change.
- Commit after each verified milestone-sized increment.

## Acceptance Criteria

1. The architecture document names real upstream files and describes the actual search and observation formats.
2. The document identifies the retriever model, index, corpus, and server requirements from source/configuration rather than memory.
3. The demo runs on the current Mac environment without a large model or GPU.
4. The demo visibly completes question, search action, observation, and answer stages.
5. No RL or future-milestone code is modified.
