# veRL / Search-R1 Compatibility Audit

The vendored Search-R1 tree is at `third_party/search-r1/`; its bundled veRL version marker is `0.1` and the repository exposes `train_grpo.sh`/`train_ppo.sh`, retrieval launch scripts, data-to-Parquet processors, and reward modules under `verl/utils/reward_score/`. The source documents GRPO, PPO, reinforce, multi-turn search, and local BM25/dense retrievers, but this milestone performs no installation or training.

The future integration points are: a custom reward callback, veRL's GRPO group advantage estimator, a multi-turn tool loop, and a query-sensitive retriever environment. `src/agentrl/verl_reward_adapter.py` is a dependency-free proof-of-interface wrapper around the shared reward core; it does not import torch, CUDA, vLLM, or ray.

The current MLX QLoRA adapter is not automatically loadable by the future PyTorch/Hugging Face veRL stack. No conversion is attempted. The safest GPU route is to reproduce SFT with the same data in a Hugging Face Qwen checkpoint, then run veRL/GRPO there. Any future MLX-to-HF conversion must be an official, output-equivalence-tested path.
