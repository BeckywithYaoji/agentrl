# Search-R1 Architecture

## Data Flow

```text
QA dataset record
↓
prompt: chat message containing question and search instructions
↓
LLM generates reasoning
↓
<search>query</search>
↓
POST /retrieve with {queries, topk, return_scores}
↓
retriever returns result: list of document lists
↓
client formats documents inside <information>...</information>
↓
prompt is extended and the LLM continues multi-turn generation
↓
<answer>...</answer>
↓
rule-based answer reward during RL
```

## Important Modules

| Module | File | Responsibility |
|---|---|---|
| Dataset | `third_party/search-r1/scripts/data_process/nq_search.py` | Converts QA data into search-training records. |
| Prompt | `third_party/search-r1/infer.py` | Instructs reasoning in `<think>`, search in `<search>`, evidence in `<information>`, and answer in `<answer>`. |
| Search action parser | `third_party/search-r1/infer.py:get_query` | Extracts the most recent search query from generated text. |
| Search client | `third_party/search-r1/search_r1/search/retrieval_request.py` | Sends batched retrieval requests to the HTTP server. |
| Search server | `third_party/search-r1/search_r1/search/retrieval_server.py` | Loads corpus/index and serves `/retrieve`. |
| Retriever | `third_party/search-r1/search_r1/search/retrieval.py` | Implements BM25 and dense retrieval backends. |
| Rollout | `third_party/search-r1/search_r1/llm_agent/generation.py` | Generates model responses and handles tool-interleaved trajectories. |
| Reward | `third_party/search-r1/verl/trainer/ppo/reward.py` | Computes rule-based answer reward. |
| RL trainer | `third_party/search-r1/verl/trainer/ppo/ray_trainer.py` | Coordinates PPO/GRPO-style training through veRL/Ray. |

## Dataset and Prompt Format

The README's custom-dataset example uses a JSON-like record with:

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "reward_model": {"ground_truth": "..."}
}
```

The question is embedded in the user message. Ground truth is stored as `reward_model.ground_truth`. `infer.py` builds a prompt that tells the model to reason inside `<think>`, optionally call search with `<search>...</search>`, consume results inside `<information>...</information>`, and finish with `<answer>...</answer>`.

## Search Protocol

Search action:

```text
<search>Hamlet author</search>
```

The parser takes the latest matching query. The HTTP client sends:

```json
{"queries": ["Hamlet author"], "topk": 3, "return_scores": true}
```

The server response is read from `result`; each query returns a list of documents. The inference client formats each document as `Doc N(Title: ...) text` and appends it as:

```text
<information>Doc 1(Title: ...) ...</information>
```

The expanded prompt is then sent to the model for another generation turn. A final answer is emitted with `<answer>...</answer>`.

## Retriever and Runtime

Search-R1 supports a separate local HTTP retriever server. BM25 uses Pyserini/Lucene; dense retrieval uses a FAISS index plus a Transformers encoder, with configuration examples using `intfloat/e5-base-v2`. The server expects a JSONL corpus whose records have `id` and `contents` fields. Online Google/SerpAPI servers are separate alternatives requiring credentials.

## Mac Compatibility

| Component | Mac M4 | Current Stage |
|---|---|---|
| Dataset preprocessing | Source-level Python; may require `datasets` | Inspect only; no download |
| Prompt/action parsing | Standard Python/regex | Reused conceptually in demo |
| HTTP search protocol | Standard HTTP/FastAPI | Reproduced locally without server |
| BM25 retriever | Requires Pyserini/Lucene setup | Not installed or run |
| Dense retriever | Upstream code calls CUDA and needs FAISS/model weights | Not suitable for this stage |
| Model inference | Requires large checkpoint and PyTorch runtime | Not run |
| veRL/Ray rollout | GPU/distributed training stack | Located only |
| PPO/GRPO | GPU/distributed training stack | Not run |

## RL Locations

The shell entry points are `third_party/search-r1/train_ppo.sh`, `third_party/search-r1/train_grpo.sh`, and the experiment scripts under `third_party/search-r1/scripts/nq_hotpotqa/`. Rollout code is under `search_r1/llm_agent/` and `verl/trainer/ppo/`; reward code is under `verl/trainer/ppo/reward.py`. None is modified or executed here.
