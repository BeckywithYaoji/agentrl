# Resume Summary

项目名称: Search-Agent Post-training for Evidence-Grounded QA

## Resume Bullets

- Built a Qwen3-0.6B search-augmented QA agent using HF/PyTorch SFT, veRL GRPO, native multi-turn `SearchXMLAgentLoop`, and TinyBM25 retrieval.
- Implemented query-sensitive retrieval evaluation, R0/R2 verifiable rewards, parser safeguards, rollout variance logging, and reward-hacking audits.
- Ran frozen Test/OOD/Counterfactual evaluation across Base, SFT, SFT+GRPO-R0, and SFT+GRPO-R2 under Train-only and Candidate BM25 protocols.
- Measured Test/Candidate BM25 EM from 0.33 Base to 0.50 SFT and 0.51 SFT+R2; archived 1.7B LoRA scaling as a negative result when 100-step LoRA-GRPO reduced val EM from 0.15 to 0.11.

## Technical Stack

Python, PyTorch, Hugging Face Transformers, PEFT/LoRA, veRL GRPO, Qwen3-0.6B, Qwen3-1.7B, SearchXMLAgentLoop, TinyBM25, JSONL/Parquet artifacts, pytest.

## Key Metrics

- 0.6B Test/Candidate BM25 EM: Base 0.33, SFT 0.50, SFT+R0 0.50, SFT+R2 0.51.
- 0.6B Test/Candidate BM25 Token F1: Base 0.4489, SFT 0.6072, SFT+R0 0.6115, SFT+R2 0.6148.
- OOD and Counterfactual Candidate BM25 EM: SFT/R0/R2 all 1.00.
- No-search OOD and Counterfactual EM: 0.00.
- 1.7B Val EM: LoRA SFT 0.15, LoRA-GRPO-R0 0.11.

## One-Line Highlight

End-to-end search-agent post-training study showing that SFT stabilized XML search behavior, while short GRPO produced small gains and retrieval coverage became the main measured bottleneck.
