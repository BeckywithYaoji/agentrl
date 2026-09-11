"""Framework-independent rewards for the Search Agent RL boundary."""
from collections import Counter
from dataclasses import dataclass
import re
import unicodedata

ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.S)
TAG_RE = re.compile(r"</?(?:search|answer)>")


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.findall(r"\w+", text, flags=re.UNICODE)


def normalized_em(prediction, truth, aliases=()):
    pred = normalize(prediction)
    return float(any(pred == normalize(candidate) for candidate in [truth, *aliases]))


def token_f1(prediction, truth, aliases=()):
    pred = normalize(prediction)
    candidates = [truth, *aliases]
    if not pred:
        return 1.0 if any(not normalize(c) for c in candidates) else 0.0
    best = 0.0
    for candidate in candidates:
        gold = normalize(candidate)
        if not gold:
            continue
        common = sum((Counter(pred) & Counter(gold)).values())
        best = max(best, 2 * common / (len(pred) + len(gold)))
    return best


@dataclass(frozen=True)
class ParsedTrajectory:
    answer: str
    actions: tuple
    protocol_valid: bool
    first_action: str | None
    search_count: int
    termination_reason: str


def parse_trajectory(raw_trajectory, termination_reason=""):
    """Parse raw assistant text or an evaluator trajectory dict."""
    if isinstance(raw_trajectory, dict):
        turns = raw_trajectory.get("turns", [])
        text = "".join(str(t.get("content", "")) for t in turns if t.get("role") == "assistant")
        termination_reason = raw_trajectory.get("termination_reason", termination_reason)
    else:
        text = str(raw_trajectory or "")
    actions = tuple((m.group(1), m.group(2).strip()) for m in ACTION_RE.finditer(text))
    malformed = bool(re.search(r"<(?:search|answer)(?:\s|>)|</(?:search|answer)", text)) and not actions
    answer_actions = [a for a in actions if a[0] == "answer"]
    valid_order = all(a[0] == "search" for a in actions[:-1]) and (not actions or actions[-1][0] == "answer")
    spans = [m.span() for m in ACTION_RE.finditer(text)]
    gaps = [text[:spans[0][0]], *[text[a[1]:b[0]] for a, b in zip(spans, spans[1:])], text[spans[-1][1]:]] if spans else [text]
    protocol_valid = bool(actions) and len(answer_actions) == 1 and valid_order and not malformed and all(not gap.strip() for gap in gaps)
    answer = answer_actions[-1][1] if answer_actions else ""
    return ParsedTrajectory(answer, actions, protocol_valid, actions[0][0] if actions else None,
                            sum(a[0] == "search" for a in actions), termination_reason)


def compute_reward(question, ground_truth, aliases=(), raw_trajectory=None,
                   requires_search=None, search_count=None, termination_reason=""):
    parsed = parse_trajectory(raw_trajectory, termination_reason)
    count = parsed.search_count if search_count is None else int(search_count)
    answer_em = normalized_em(parsed.answer, ground_truth, aliases)
    answer_f1 = token_f1(parsed.answer, ground_truth, aliases)
    decision_correct = None if requires_search is None else float(
        parsed.first_action == ("search" if requires_search else "answer"))
    ideal = 1 if requires_search else 0
    excess_searches = max(0, count - ideal)
    r0 = answer_em
    r1 = answer_em * float(parsed.protocol_valid)
    if not answer_em:
        r2 = -0.1 if not parsed.protocol_valid else 0.0
    else:
        r2 = 1.0
        if not parsed.protocol_valid:
            r2 -= 0.20
        if decision_correct is not None and not decision_correct:
            r2 -= 0.10
        r2 -= min(0.10, 0.02 * excess_searches)
        r2 = max(-0.1, min(1.0, r2))
    return {
        "reward": {"r0": r0, "r1": r1, "r2": r2},
        "answer": parsed.answer, "answer_em": answer_em, "answer_f1": answer_f1,
        "protocol_valid": int(parsed.protocol_valid), "decision_correct": decision_correct,
        "first_action": parsed.first_action, "search_count": count,
        "excess_searches": excess_searches, "termination_reason": parsed.termination_reason,
    }


def reward_summary(rows):
    values = {name: [float(r["reward"][name]) for r in rows] for name in ("r0", "r1", "r2")}
    out = {}
    for name, vals in values.items():
        mean = sum(vals) / len(vals) if vals else 0.0
        ordered = sorted(vals)
        median = ordered[len(ordered)//2] if len(ordered) % 2 else (ordered[len(ordered)//2-1] + ordered[len(ordered)//2]) / 2 if ordered else 0.0
        out[name] = {"mean": mean, "median": median,
                     "std": (sum((x-mean)**2 for x in vals) / len(vals))**0.5 if vals else 0.0,
                     "min": min(vals) if vals else 0.0, "max": max(vals) if vals else 0.0,
                     "positive_ratio": sum(x > 0 for x in vals)/len(vals) if vals else 0.0,
                     "zero_ratio": sum(x == 0 for x in vals)/len(vals) if vals else 0.0,
                     "negative_ratio": sum(x < 0 for x in vals)/len(vals) if vals else 0.0}
    return out
