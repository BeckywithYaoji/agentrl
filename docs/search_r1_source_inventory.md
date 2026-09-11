# Search-R1 Source Inventory

The upstream source is preserved under `third_party/search-r1/` from the official `main` branch archive. The Git clone transport timed out in this environment, so this vendored snapshot does not contain upstream `.git` metadata; the source URL is `https://github.com/PeterGriffinJin/Search-R1`.

## Provenance

- Official repository: `https://github.com/PeterGriffinJin/Search-R1`
- Acquisition method: GitHub `main` branch tarball, because Git clone timed out
- Branch: `main`
- Upstream commit: `UNKNOWN` (the downloaded archive did not include Git metadata)
- Acquisition date: `2026-09-11` (local date)
- Original tarball SHA256: `UNKNOWN` (the transient download was not retained)
- Current vendored-directory tarball snapshot SHA256: `59ca833d0e57bd224b754870b2b58596d034a8324b57cbe106fde350f4bf9433`

| Area | Source file | Finding |
|---|---|---|
| Dataset preprocessing | `scripts/data_process/nq_search.py`, `scripts/data_process/nq.py` | Converts QA records into prompt/reward records; `nq_search.py` adds search-oriented prompts. |
| Prompt schema | `README.md` (Use your own dataset), `infer.py` | A record contains `prompt` as chat messages and `reward_model.ground_truth`; inference uses explicit `<think>`, `<search>`, `<information>`, and `<answer>` tags. |
| Search action parsing | `infer.py:get_query` | Extracts the latest text matched by `<search>(.*?)</search>` with DOTALL. |
| Retrieval API | `search_r1/search/retrieval_request.py` | POSTs `queries`, `topk`, and `return_scores` to `/retrieve`; expects a `result` field. |
| Retriever server | `search_r1/search/retrieval_server.py` | Loads a corpus and BM25/dense retriever, then exposes `/retrieve`. |
| RL rollout | `search_r1/llm_agent/generation.py`, `verl/trainer/ppo/ray_trainer.py` | Generates trajectories and integrates search calls into rollout/training. |
| Reward | `search_r1/llm_agent/generation.py`, `verl/trainer/ppo/reward.py` | Uses answer extraction and rule-based outcome reward in the training path. |
| RL entry points | `train_ppo.sh`, `train_grpo.sh`, `scripts/nq_hotpotqa/v0.*/train_*.sh` | Shell entry points for PPO/GRPO experiments; not run in this milestone. |

## Environment findings

The reference inference script requires Transformers, PyTorch, a model checkpoint, and a running retriever service. The upstream dense retriever server calls `.cuda()` and depends on FAISS/Transformers, so it is not a suitable direct runtime for this Mac-only smoke test. The demo therefore reproduces the observed request/response protocol with a fixed in-memory corpus and no third-party runtime dependency.
