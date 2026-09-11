"""Thin CPU-safe adapter for Search-R1/veRL-style reward callbacks."""
from .reward import compute_reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    extra_info = extra_info or {}
    result = compute_reward(
        question=extra_info.get("question", ""),
        ground_truth=ground_truth,
        aliases=extra_info.get("aliases", ()),
        raw_trajectory=solution_str,
        requires_search=extra_info.get("requires_search"),
        search_count=extra_info.get("search_count"),
        termination_reason=extra_info.get("termination_reason", ""),
    )
    return {"score": result["reward"]["r2"], "reward": result, "data_source": data_source}
