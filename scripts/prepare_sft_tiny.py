"""Deterministic train-only 32-target learning probe."""
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agentrl.sft_dataset import read_jsonl, write_jsonl


def main():
    rng=random.Random(42)
    canonical={r['id']:r for r in read_jsonl('data/sft/train.jsonl')}
    rows=read_jsonl('artifacts/milestone4_audit/data/train.jsonl')
    specs=[('direct_answer',1,4),('single_search_clean',1,4),
           ('single_search_hard_negative',1,4),('single_search_clean',3,8),
           ('single_search_hard_negative',3,8),('noisy_retrieval',3,4)]
    chosen=[]
    for task,index,n in specs:
        candidates=[r for r in rows if canonical[r['canonical_id']]['task_type']==task and r['target_message_index']==index]
        chosen.extend(rng.sample(candidates,n))
    rng.shuffle(chosen)
    valid=rng.sample(read_jsonl('artifacts/milestone4_audit/data/valid.jsonl'),8)
    write_jsonl('data/sft_tiny/train.jsonl',chosen)
    write_jsonl('data/sft_tiny/valid.jsonl',valid)
    print('32 train-only targets; 8 original-validation targets; no counterfactual-test data used.')


if __name__=='__main__':
    main()
