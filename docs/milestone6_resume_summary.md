# Milestone 6 Resume Summary

项目名称: SearchXML Agent RL for Evidence-Grounded QA

## Resume Bullets

- Built a Qwen3-0.6B SearchXML QA agent with HF/PyTorch SFT, veRL GRPO, native multi-turn tool use, and TinyBM25 retrieval.
- Implemented R0 and R2 reward evaluation with parser safeguards, reward-component logging, and reward-hacking audits across 100-step GRPO runs.
- Ran frozen Test/OOD/Counterfactual evaluation across Base, SFT, SFT+R0, and SFT+R2 under Train-only and Candidate BM25 protocols.
- Measured Test/Candidate EM from 0.33 Base to 0.50 SFT and 0.51 SFT+R2; SFT/R0/R2 reached 1.00 EM on OOD and Counterfactual Candidate BM25.

## 技术栈

Python, PyTorch, Hugging Face Transformers, veRL GRPO, Qwen3-0.6B, SearchXMLAgentLoop, TinyBM25, pytest, JSONL/Parquet evaluation artifacts.

## 关键指标

- Test/Candidate BM25 EM: Base 0.33, SFT 0.50, SFT+R0 0.50, SFT+R2 0.51.
- OOD/Candidate BM25 EM: SFT/R0/R2 1.00.
- Counterfactual/Candidate BM25 EM: SFT/R0/R2 1.00.
- Frozen leakage audit: train overlap with Test/OOD/Counterfactual = 0/0/0.

## 项目亮点

A compact end-to-end agent RL study showing that SFT stabilizes SearchXML tool use, while retrieval coverage dominates held-out performance when evidence is absent from the train-only corpus.
