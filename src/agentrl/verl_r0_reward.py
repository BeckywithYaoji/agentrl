"""R0 reward adapter: exact match on the final XML answer only."""
from agentrl.reward import normalized_em, parse_trajectory

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    parsed = parse_trajectory(solution_str)
    if not parsed.protocol_valid or not parsed.answer:
        # A valid answer action is required, but protocol diagnostics do not add reward.
        return normalized_em(parsed.answer, ground_truth)
    return normalized_em(parsed.answer, ground_truth)
