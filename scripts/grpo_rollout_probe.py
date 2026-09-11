#!/usr/bin/env python3
"""Validation-only stochastic group rollout probe; never trains."""
import argparse, json, random, statistics
from pathlib import Path
from agentrl.qwen_agent import QwenAgent
from agentrl.agent_loop import run_agent
from agentrl.reward import compute_reward
from agentrl.sft_evaluation import ReplayRetriever


class StochasticAgent:
    def __init__(self, adapter, temperature, top_p):
        self.agent=QwenAgent(max_tokens=256, adapter_path=adapter)
        self.temperature, self.top_p=temperature, top_p
    def generate(self, messages):
        return self.agent.generate(messages, temperature=self.temperature, top_p=self.top_p)


def load(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def pick(rows, seed=42):
    buckets={}
    for row in rows:
        buckets.setdefault(row.get("task_type", "other"), []).append(row)
    rng=random.Random(seed)
    selected=[]
    preferred=["direct_answer", "single_search_clean", "single_search_hard_negative", "insufficient_evidence"]
    for name in preferred:
        candidates=buckets.get(name, [])
        rng.shuffle(candidates); selected.extend(candidates[:6])
    if len(selected)<24:
        rest=[r for name, values in buckets.items() if name not in preferred for r in values]
        rng.shuffle(rest); selected.extend(rest[:24-len(selected)])
    return selected[:24]


def summarize(groups):
    result={"group_count":len(groups), "rollouts_per_group":4, "rewards":{}}
    for name in ("r0","r1","r2"):
        vals=[v for g in groups for v in g[name]]
        result["rewards"][name]={"mean":statistics.mean(vals) if vals else 0.0, "std":statistics.pstdev(vals) if vals else 0.0,
            "zero_variance_group_rate":sum(statistics.pstdev(g[name])==0 for g in groups)/len(groups) if groups else 0.0,
            "all_zero_group_rate":sum(all(v==0 for v in g[name]) for g in groups)/len(groups) if groups else 0.0,
            "all_perfect_group_rate":sum(all(v==1 for v in g[name]) for g in groups)/len(groups) if groups else 0.0}
    rows=[r for g in groups for r in g["details"]]
    result.update(protocol_success=sum(r["success"] for r in rows)/len(rows), answer_em=sum(r["reward"]["answer_em"] for r in rows)/len(rows),
                  search_decision_accuracy=sum(r["reward"]["decision_correct"] for r in rows if r["reward"]["decision_correct"] is not None)/sum(r["reward"]["decision_correct"] is not None for r in rows),
                  average_search_count=statistics.mean(r["search_count"] for r in rows))
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--valid", default="data/sft/val.jsonl"); ap.add_argument("--adapter", required=True); ap.add_argument("--output", default="artifacts/milestone5a/grpo_probe.jsonl"); ap.add_argument("--summary", default="artifacts/milestone5a/grpo_probe_summary.json")
    ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--top-p", type=float, default=0.95)
    args=ap.parse_args(); rows=pick(load(args.valid)); assert len(rows)==24
    agent=StochasticAgent(args.adapter,args.temperature,args.top_p); groups=[]
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w") as f:
        for i,sample in enumerate(rows):
            docs=sample["metadata"]["documents"]; details=[]; rewards=[]
            for k in range(4):
                result=run_agent(sample["question"], agent, ReplayRetriever(docs), max_search_steps=3)
                scored=compute_reward(sample["question"], sample["answer"], sample["metadata"].get("aliases",()), result, sample["metadata"].get("requires_search"), result["search_count"], result["termination_reason"])
                result.update(sample_id=sample["id"], group_index=i, rollout_index=k, task_type=sample.get("task_type"), reward=scored)
                f.write(json.dumps(result,ensure_ascii=False)+"\n"); details.append(result); rewards.append(scored["reward"])
            groups.append({"sample_id":sample["id"], "task_type":sample.get("task_type"), **{name:[r[name] for r in rewards] for name in ("r0","r1","r2")}, "details":details})
            print(i+1, sample["id"], {n:[r[n] for r in rewards] for n in ("r0","r1","r2")})
    summary=summarize(groups); summary.update(policy="SFT", validation_only=True, temperature=args.temperature, top_p=args.top_p, groups=[{k:v for k,v in g.items() if k!="details"} for g in groups])
    Path(args.summary).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__ == "__main__": main()
