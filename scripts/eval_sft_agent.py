"""Same frozen evaluator for base and SFT. Raw outputs are never overwritten."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agentrl.sft_dataset import read_jsonl
from src.agentrl.sft_evaluation import EVAL_CONFIG, evaluate_case, summarize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',choices=['base','sft'],required=True)
    parser.add_argument('--adapter',type=Path)
    args=parser.parse_args()
    if (args.label=='sft') != bool(args.adapter):
        parser.error('SFT requires an adapter; base must not have one')
    out=Path('artifacts/milestone4_eval'); out.mkdir(parents=True,exist_ok=True)
    sources={'m2b':Path('data/eval/milestone2b_baseline.jsonl'),
             'test':Path('data/sft/test.jsonl'), 'counterfactual':Path('data/sft/counterfactual.jsonl'),
             'ood':Path('data/sft/test_ood.jsonl')}
    for name in sources:
        if (out/f'{args.label}_{name}.jsonl').exists():
            parser.error('raw output already exists; refusing overwrite')
    from src.agentrl.qwen_agent import QwenAgent,MODEL,REVISION
    agent=QwenAgent(max_tokens=EVAL_CONFIG['max_new_tokens'],adapter_path=str(args.adapter) if args.adapter else None)
    manifest={'model':MODEL,'revision':REVISION,'config':EVAL_CONFIG,'label':args.label,
              'adapter_sha256':hashlib.sha256((args.adapter/'adapters.safetensors').read_bytes()).hexdigest() if args.adapter else None,
              'dataset_sha256':{n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sources.items()}, 'results':{}}
    for name,path in sources.items():
        rows=[]
        with (out/f'{args.label}_{name}.jsonl').open('x') as file:
            for sample in read_jsonl(path):
                row=evaluate_case(sample,agent,smoke=name=='m2b')
                file.write(json.dumps(row,ensure_ascii=False)+'\n'); file.flush()
                rows.append(row)
                print(f"{args.label} {name} {len(rows)} {row['id']} {row['termination_reason']} EM={row['answer_em']}",flush=True)
        subsets={t:summarize([r for r in rows if r['task_type']==t]) for t in sorted({r['task_type'] for r in rows})}
        manifest['results'][name]={'overall':summarize(rows),'subsets':subsets}
        (out/f'{args.label}_summary.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
