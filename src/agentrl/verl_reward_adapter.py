"""Thin CPU-safe adapter for Search-R1/veRL-style reward callbacks."""
from agentrl.reward import compute_reward


def reward_extra_info(sample):
    """Only task intent metadata crosses from a train row into R2 scoring."""
    required = sample["metadata"]["requires_search"]
    if not isinstance(required, bool):
        raise ValueError("requires_search must be a bool")
    return {"question": sample["question"], "requires_search": required}


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    extra_info = extra_info or {}
    if "requires_search" not in extra_info or not isinstance(extra_info["requires_search"], bool):
        raise ValueError("requires_search is required for R2 decision scoring")
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
