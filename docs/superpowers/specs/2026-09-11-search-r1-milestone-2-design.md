# Search-R1 Milestone 2 Design

## Goal

Run one real Qwen3-on-MLX multi-turn trajectory on Apple Silicon: model output → Search-R1 search action → deterministic mock observation → model continuation → final answer.

## Scope and constraints

Only local inference is in scope. Use MLX-LM and the smallest usable Qwen3 model, prefer Qwen3-0.6B, reuse the existing Search-R1 parser and deterministic mock knowledge base, and cap search turns at three. Do not add training, web search, CUDA, vLLM, veRL execution, reward models, or large indexes.

## Components

- `src/agentrl/qwen_agent.py`: thin MLX-LM generation adapter.
- `scripts/qwen_smoke_test.py`: model-only generation check.
- `scripts/qwen_search_demo.py`: multi-turn loop, tool budget, trajectory JSONL, and smoke metrics.
- existing `scripts/minimal_search_demo.py`: source of the parser and local retrieval behavior; no duplicate parser.
- tests use `FakeAgent` and never load model weights.

## Acceptance

The real model must load and generate on the M4. The integration result must honestly report whether it produced a valid search action and final answer; unit tests must prove search parsing, observation append, answer termination, budget enforcement, and trajectory schema. Counterfactual success is measured, not assumed.
