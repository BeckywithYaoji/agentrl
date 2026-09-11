# RL Environment Contract

Milestone 4's `ReplayRetriever` is a controlled evaluation/probe environment. It returns documents by sample ID regardless of the query. It must not be used for final Search Agent GRPO training: a garbage query can still receive gold evidence and create a query-quality reward exploit.

The future training environment must implement `search(query) -> documents`. Query changes should be able to change returned documents; `asdfghjkl` must not reliably retrieve gold evidence; the environment must never inspect the ground-truth answer to choose documents; and the reward must not read a gold-document position. Evidence is an observation, not a direct reward.

The recommended first GPU environment is a query-sensitive lightweight retriever over the controlled corpus. Search-R1's BM25 path is CPU-friendly and simple; its dense E5 path may improve semantic recall but adds model/index cost and can use GPU or CPU ANN. Do not download Wikipedia or combine Qwen, veRL, vLLM, GRPO, and an open-domain index in the first integration.
