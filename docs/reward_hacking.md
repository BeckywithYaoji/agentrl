# Reward Hacking Risks

| Exploit | Cause | Detection | Mitigation |
|---|---|---|---|
| Format farming | Positive format reward independent of outcome | Wrong-answer perfect-format test | R1 gates outcome; R2 is correctness-dominant |
| Search everything | Search trigger rewarded | Direct-answer subset | Decision metric and small cost tie-breaker |
| Garbage query | Oracle replay returns gold evidence | `asdfghjkl` query test | Query-sensitive training retriever |
| Distractor copying | Evidence overlap rewarded | Hard-negative and distractor tests | Outcome verification only |
| Excessive search | No cost or budget | Redundant-search test | Small efficiency term and max-step budget |
| Answer stuffing | Substring metric | `Paris London` vs `Paris` | Normalized exact match |

Search cost is an efficiency regularizer only. It is deliberately too small to make a correct searched answer worse than an incorrect direct answer.
