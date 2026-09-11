#!/usr/bin/env python3
"""Offline R0/R1/R2 audit for saved Base/SFT trajectories."""
import argparse, json, statistics
from pathlib import Path
from agentrl.reward import compute_reward, reward_summary


def load(path):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def audit(path):
    rows=[]
    for sample in load(path):
        r=compute_reward(sample.get("question", ""), sample.get("expected_answer", ""),
                         sample.get("aliases", ()), sample,
                         sample.get("requires_search"), sample.get("search_count"),
                         sample.get("termination_reason", ""))
        r["id"] = sample.get("id")
        r["correct"] = bool(sample.get("answer_em", r["answer_em"]))
        rows.append(r)
    correct=[r for r in rows if r["correct"]]
    wrong=[r for r in rows if not r["correct"]]
    result={"path":str(path), "count":len(rows), "rewards":reward_summary(rows),
            "reward_correct_mean": {k:statistics.mean([r["reward"][k] for r in correct]) if correct else None for k in ("r0","r1","r2")},
            "reward_wrong_mean": {k:statistics.mean([r["reward"][k] for r in wrong]) if wrong else None for k in ("r0","r1","r2")},
            "protocol_valid_rate":sum(r["protocol_valid"] for r in rows)/len(rows) if rows else 0,
            "decision_correct_rate":sum(r["decision_correct"] for r in rows if r["decision_correct"] is not None)/sum(r["decision_correct"] is not None for r in rows) if any(r["decision_correct"] is not None for r in rows) else None}
    return result, rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--output", required=True)
    args=ap.parse_args(); reports=[]
    for path in args.paths:
        report,_=audit(path); reports.append(report); print(json.dumps(report, ensure_ascii=False))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(reports, indent=2, ensure_ascii=False)+"\n")

if __name__ == "__main__": main()
