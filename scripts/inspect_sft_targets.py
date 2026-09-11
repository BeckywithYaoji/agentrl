"""Stage 4A: audit actual target masks using the installed MLX batching code."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_mlx_sft import export_rows
from src.agentrl.sft_dataset import read_jsonl, write_jsonl
from src.agentrl.sft_training import tokenize_target, supervised_positions, TargetDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/milestone4_audit'))
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from mlx_lm.tuner.trainer import iterate_batches
    from src.agentrl.qwen_agent import MODEL, REVISION
    config = hf_hub_download(MODEL, 'tokenizer_config.json', revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(Path(config).parent), local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'model': MODEL, 'revision': REVISION, 'max_seq_length': 1536, 'splits': {}, 'inspected': []}
    exported = {}
    canonical = {}
    for source, destination in [('train', 'train'), ('val', 'valid')]:
        rows = read_jsonl(f'data/sft/{source}.jsonl')
        canonical.update({r['id']: r for r in rows})
        exported[destination] = export_rows(rows)
        write_jsonl(args.output / 'data' / f'{destination}.jsonl', exported[destination])
        lengths = sorted(len(tokenize_target(r, tokenizer)[0]) for r in exported[destination])
        def quantile(p):
            return lengths[round((len(lengths)-1)*p)]
        summary['splits'][destination] = dict(count=len(lengths), min=lengths[0], median=quantile(.5),
            p90=quantile(.9), p95=quantile(.95), p99=quantile(.99), max=lengths[-1],
            truncated=sum(n > 1536 for n in lengths),
            sha256=hashlib.sha256((args.output/'data'/f'{destination}.jsonl').read_bytes()).hexdigest())
    rng = random.Random(42)
    train = exported['train']
    groups = {
        'direct': [r for r in train if canonical[r['canonical_id']]['task_type']=='direct_answer'],
        'search': [r for r in train if r['messages'][-1]['content'].startswith('<search>')],
        'evidence_answer': [r for r in train if r['target_message_index']==3],
    }
    for kind, rows in groups.items():
        row = rng.choice(rows)
        tokens, offset = tokenize_target(row, tokenizer)
        dataset = TargetDataset([row], tokenizer)
        batch, lengths = next(iterate_batches(dataset, batch_size=1, max_seq_length=1536, loop=False))
        padded = batch.shape[1]
        actual_offset, actual_length = lengths.tolist()[0]
        indices = supervised_positions(actual_offset, actual_length, padded)
        supervised = tokenizer.decode([batch[0, i].item() for i in indices])
        assert indices == list(range(offset, len(tokens)))
        assert supervised.startswith(row['messages'][-1]['content'])
        assert '<information>' not in supervised and '<think>' not in supervised
        item = dict(kind=kind, canonical_id=row['canonical_id'], context=row['messages'][:-1],
            target=row['messages'][-1]['content'], total_tokens=len(tokens), masked_tokens=offset,
            supervised_tokens=len(indices), decoded_supervised=supervised,
            default_loss_extra_indices=[i for i in range(1,padded) if offset <= i <= len(tokens) and i not in indices])
        summary['inspected'].append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)
    assert all(s['truncated']==0 for s in summary['splits'].values())
    summary['result'] = 'PASS'
    (args.output/'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary['splits'], indent=2), flush=True)


if __name__ == '__main__':
    main()
