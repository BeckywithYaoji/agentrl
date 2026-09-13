"""Unified frozen split evaluation utilities for Milestone 6-F."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from agentrl.agent_loop import run_agent
from agentrl.hf_sft import RUNTIME_SYSTEM_PROMPT
from agentrl.retrieval import TinyBM25Retriever, load_training_corpus
from agentrl.reward import normalized_em, parse_trajectory, token_f1


LOCKED_EVAL_CONFIG = {
    "system_prompt": RUNTIME_SYSTEM_PROMPT,
    "enable_thinking": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "max_new_tokens": 256,
    "max_search_steps": 3,
    "top_k": 3,
    "normalization": "agentrl.reward normalized_em/token_f1",
}

FROZEN_SPLITS = {
    "test": "data/sft/test.jsonl",
    "ood": "data/sft/test_ood.jsonl",
    "counterfactual": "data/sft/counterfactual.jsonl",
}


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", str(text or "")).casefold())


def _doc_contents(document: Mapping[str, str]) -> str:
    return document["title"] + "\n" + document["text"]


class TrainOnlyBM25Protocol:
    """BM25 over the training corpus only."""

    name = "train_only_bm25"

    def __init__(self, train_path: str | Path = "data/sft/train.jsonl") -> None:
        self.train_path = str(train_path)
        self.retriever = TinyBM25Retriever(load_training_corpus(train_path))

    def retriever_for(self, sample: Mapping) -> TinyBM25Retriever:
        return self.retriever

    def diagnostic_for(self, sample: Mapping, queries: Iterable[str], topk: int) -> dict:
        return {"held_out_evidence_coverage": 0.0}


class CandidateBM25Retriever:
    """Per-sample query-sensitive BM25 over candidate document text only."""

    def __init__(self, documents: Iterable[Mapping[str, object]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = tuple(
            {
                "id": str(document["id"]),
                "title": str(document["title"]),
                "text": str(document["text"]),
            }
            for document in documents
        )
        self.contents = tuple(_doc_contents(document) for document in self.documents)
        self._frequencies = tuple(Counter(_tokens(text)) for text in self.contents)
        self._lengths = tuple(sum(frequency.values()) for frequency in self._frequencies)
        count = len(self.contents)
        self._avgdl = sum(self._lengths) / count if count else 0.0
        document_frequency = Counter()
        for frequencies in self._frequencies:
            document_frequency.update(frequencies.keys())
        self._idf = {
            term: math.log1p((count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def ranked_ids(self, query: str, topk: int = 3) -> list[str]:
        if topk <= 0 or not self._avgdl:
            return []
        terms = _tokens(query)
        scored = []
        for index, frequencies in enumerate(self._frequencies):
            norm = self.k1 * (1 - self.b + self.b * self._lengths[index] / self._avgdl)
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self._idf[term] * frequency * (self.k1 + 1) / (frequency + norm)
            if score > 0:
                scored.append((index, score))
        scored.sort(key=lambda item: -item[1])
        return [self.documents[index]["id"] for index, _ in scored[:topk]]

    def retrieve(self, queries: list[str], topk: int = 3) -> dict:
        rows = []
        by_id = {document["id"]: document for document in self.documents}
        for query in queries:
            rows.append([
                {"document": {"contents": _doc_contents(by_id[document_id]), "id": document_id}}
                for document_id in self.ranked_ids(query, topk)
            ])
        return {"result": rows}


class FrozenCandidateBM25Protocol:
    """BM25 over each sample's frozen metadata.documents candidate pool."""

    name = "candidate_bm25"

    def retriever_for(self, sample: Mapping) -> CandidateBM25Retriever:
        return CandidateBM25Retriever(sample["metadata"].get("documents", ()))

    def support_ids(self, sample: Mapping) -> set[str]:
        return {
            str(document["id"])
            for document in sample["metadata"].get("documents", ())
            if document.get("is_support") is True
        }

    def diagnostic_for(self, sample: Mapping, queries: Iterable[str], topk: int) -> dict:
        retriever = self.retriever_for(sample)
        supports = self.support_ids(sample)
        searched = [query for query in queries if query]
        hit1 = hit3 = 0
        for query in searched:
            ranking = retriever.ranked_ids(query, topk=max(3, topk))
            hit1 += bool(ranking[:1] and supports.intersection(ranking[:1]))
            hit3 += bool(supports.intersection(ranking[:3]))
        garbage = retriever.ranked_ids("zzzzzz_nonmatching_query_token", topk=max(3, topk))
        return {
            "retrieval_hit_at_1": hit1 / len(searched) if searched else None,
            "retrieval_hit_at_3": hit3 / len(searched) if searched else None,
            "garbage_support_hit": bool(supports.intersection(garbage)),
            "support_count": len(supports),
        }


def evaluate_sample(sample: Mapping, agent, protocol, config: Mapping = LOCKED_EVAL_CONFIG) -> dict:
    retriever = protocol.retriever_for(sample)
    result = run_agent(
        sample["question"],
        agent,
        retriever,
        max_search_steps=int(config["max_search_steps"]),
    )
    parsed = parse_trajectory(result)
    answer = result["final_answer"] or ""
    queries = [turn.get("query", "") for turn in result["turns"] if turn.get("role") == "tool"]
    diagnostic = protocol.diagnostic_for(sample, queries, int(config["top_k"]))
    requires_search = bool(sample.get("metadata", {}).get("requires_search"))
    first_action = parsed.first_action
    record = {
        "id": sample["id"],
        "task_type": sample.get("task_type"),
        "answer": answer,
        "expected_answer": sample["answer"],
        "answer_em": normalized_em(answer, sample["answer"], sample.get("aliases", ())) if answer else 0.0,
        "token_f1": token_f1(answer, sample["answer"], sample.get("aliases", ())),
        "protocol_valid": parsed.protocol_valid,
        "first_action": first_action,
        "requires_search": requires_search,
        "search_triggered": result["search_count"] > 0,
        "search_count": result["search_count"],
        "termination_reason": result["termination_reason"],
        "queries": queries,
        "turns": result["turns"],
        **diagnostic,
    }
    if sample.get("task_type") == "counterfactual":
        record["counterfactual_em"] = record["answer_em"]
        record["evidence_use_success"] = float(record["answer_em"] and record["search_triggered"])
    return record


def summarize_records(records: list[Mapping]) -> dict:
    n = len(records)
    if not n:
        return {"count": 0}
    decision_rows = [row for row in records if isinstance(row.get("requires_search"), bool)]
    tp = sum(row["first_action"] == "search" and row["requires_search"] for row in decision_rows)
    fp = sum(row["first_action"] == "search" and not row["requires_search"] for row in decision_rows)
    fn = sum(row["first_action"] != "search" and row["requires_search"] for row in decision_rows)
    direct = sum(not row["requires_search"] for row in decision_rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    summary = {
        "count": n,
        "answer_em": sum(row["answer_em"] for row in records) / n,
        "token_f1": sum(row["token_f1"] for row in records) / n,
        "protocol_success": sum(row["protocol_valid"] for row in records) / n,
        "search_trigger_rate": sum(row["search_triggered"] for row in records) / n,
        "search_decision_accuracy": (
            sum(row["first_action"] == ("search" if row["requires_search"] else "answer") for row in decision_rows)
            / len(decision_rows)
            if decision_rows
            else None
        ),
        "search_precision": precision,
        "search_recall": recall,
        "search_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "avg_searches": sum(row["search_count"] for row in records) / n,
        "max_step_failure_rate": sum(row["termination_reason"] == "max_search_steps" for row in records) / n,
        "direct_search_false_positive_rate": fp / direct if direct else None,
    }
    hit1 = [row["retrieval_hit_at_1"] for row in records if row.get("retrieval_hit_at_1") is not None]
    hit3 = [row["retrieval_hit_at_3"] for row in records if row.get("retrieval_hit_at_3") is not None]
    if hit1:
        summary["retrieval_hit_at_1"] = sum(hit1) / len(hit1)
        summary["retrieval_hit_at_3"] = sum(hit3) / len(hit3)
    if any(row.get("task_type") == "counterfactual" for row in records):
        counterfactual_rows = [row for row in records if row.get("task_type") == "counterfactual"]
        summary["counterfactual_em"] = sum(row.get("counterfactual_em", 0.0) for row in counterfactual_rows) / len(counterfactual_rows)
        summary["evidence_use_success"] = sum(row.get("evidence_use_success", 0.0) for row in counterfactual_rows) / len(counterfactual_rows)
    return summary


def leakage_audit(split_paths: Mapping[str, str] = FROZEN_SPLITS, train_path: str | Path = "data/sft/train.jsonl") -> dict:
    train_ids = {row["id"] for row in read_jsonl(train_path)}
    overlaps = {
        split: sorted(train_ids.intersection(row["id"] for row in read_jsonl(path)))
        for split, path in split_paths.items()
    }
    return {
        "train_path": str(train_path),
        "split_paths": dict(split_paths),
        "train_id_overlap_counts": {split: len(ids) for split, ids in overlaps.items()},
        "overlap_ids": overlaps,
        "frozen_splits_not_used_for_training": all(not ids for ids in overlaps.values()),
    }


def candidate_bm25_audit(samples: Iterable[Mapping]) -> dict:
    changed = 0
    garbage_hits = 0
    checked = 0
    for sample in samples:
        documents = sample.get("metadata", {}).get("documents", [])
        if not documents:
            continue
        retriever = CandidateBM25Retriever(documents)
        support_ids = {str(document["id"]) for document in documents if document.get("is_support") is True}
        first = retriever.ranked_ids(documents[0]["title"], topk=3)
        last = retriever.ranked_ids(documents[-1]["title"], topk=3)
        changed += first != last
        garbage_hits += bool(support_ids.intersection(retriever.ranked_ids("zzzzzz_nonmatching_query_token", topk=3)))
        checked += 1
    return {
        "candidate_samples_checked": checked,
        "query_sensitive_rate": changed / checked if checked else None,
        "garbage_support_hit_rate": garbage_hits / checked if checked else None,
        "ranking_uses_support_labels": False,
        "ranking_uses_answers": False,
    }
