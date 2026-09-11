# Milestone 5B Stack Decision

## Host audit

Audit date: 2026-09-11. Host: Apple Silicon arm64 macOS Darwin 25.5.0, Python 3.14.6. `nvidia-smi` and `nvcc` are unavailable, so NVIDIA GPU model, VRAM, driver, and CUDA toolkit are not present on this host. Docker Desktop is available (`29.5.2` server, Ubuntu 24.04.4 aarch64), but its runtime list has no NVIDIA runtime.

The audited host environment has no importable `torch`, `transformers`, `vllm`, `verl`, `ray`, or `sglang`. No packages were installed or upgraded.

## Decision

No GPU stack is selected or executed in this milestone. The required GPU gate fails before environment creation. Neither the vendored Search-R1-compatible stack nor a current veRL stack can honestly satisfy Qwen3 HF generation, vLLM/SGLang rollout, backward, and optimizer-step requirements on this host.

When a real NVIDIA host is provided, select exactly one isolated stack after repeating this audit. Prefer a pinned current veRL stack with native Qwen3, GRPO, and multi-turn tool-loop support; use the vendored Search-R1 stack only if its exact pinned dependencies are compatible with that host. Do not mix both stacks or install `latest` packages.

## Gate result

`FAIL / BLOCKED: no NVIDIA GPU or CUDA host available.` No HF model download, vLLM/SGLang server, retriever training corpus, standalone GPU loop, veRL rollout, or GRPO optimizer step was run. This is an environment blocker, not a model or reward failure.
