# Search Agent SFT Dataset — Milestone 3

## Scope and design

Data construction only; no training or model generation. The initial heuristic distribution
is 150 direct arithmetic, 400 clean search, 300 same-topic hard-negative search,
100 noisy search and 50 insufficient-evidence trajectories. Multi-search is omitted
(0%, within the <=10% limit) to prioritize single-search reliability. Two independent
30-row synthetic evaluation sets cover counterfactual grounding and OOD entities/templates.

## Sources and license

SQuAD 2.0 by Pranav Rajpurkar, Robin Jia and Percy Liang (2018), based on Wikipedia
and crowdworker question/answer annotations. Official source:
https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

The [official dataset website](https://rajpurkar.github.io/SQuAD-explorer/) distributes
the dataset under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Our derived JSONL data is distributed under the same license; credit the SQuAD authors
and Wikipedia contributors. Original article titles, paragraph IDs, QA IDs, answer spans,
URL and file SHA256 are retained for attribution and reproducibility. Modifications:
subsampling, title-prefixed questions, query transformation, evidence shuffling and
composition, protocol messages and local data splits. Synthetic arithmetic/registry
fixtures are also released under CC BY-SA 4.0. This data notice does not relicense code.
The 40 MB original source is cached outside the repository and is not committed.

Search-R1's `scripts/nq_hotpotqa/data_process.sh` uses NQ and HotpotQA for training,
and NQ/TriviaQA/PopQA/HotpotQA/2WikiMultihopQA/MuSiQue/Bamboogle for evaluation.
`qa_search_train_merge.py` reads FlashRAG `golden_answers` into
`reward_model.ground_truth.target`. These pipelines lack the per-document support
annotation needed here; SQuAD supplies actual source paragraphs and answer offsets.

Inspection of `search_r1/llm_agent/generation.py` and `verl/utils/reward_score/qa_em.py`
found search/answer actions and answer matching, but no dedicated insufficient-evidence
policy. `<answer>Insufficient information.</answer>` is an explicit project extension,
not claimed to be upstream behavior. It is used only for SQuAD's native `is_impossible`
items with their original paragraph and empty gold-answer list.

## Development gates

The ten-row gate covers two direct, two clean, two hard negative, one noisy, one
insufficient and two counterfactual records. It passes protocol/schema validation,
duplicate checks and statistics; 21 unit tests pass including all previous milestones.
The implementation now constructs only the requested tiny rows before scaling.
Next gates are 100 then 1000 core trajectories.
