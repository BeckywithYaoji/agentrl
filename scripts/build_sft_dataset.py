"""Build and validate small canonical datasets from a local official SQuAD file."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agentrl.sft_dataset import (SOURCE_URL, LICENSE, build_main, counterfactual_set,
    load_source, validate_splits, write_jsonl, statistics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, choices=[10, 100, 1000], default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be empty; use a fresh directory to preserve prior stages')
    source = load_source(args.source)
    if args.size == 10:
        # Select 8 source/direct rows plus 2 counterfactuals, across all required tiny types.
        rows = build_main(source, 10, args.seed)['tiny']
        rows.extend(counterfactual_set(2, args.seed))
        splits = {'tiny': rows}
    else:
        splits = build_main(source, args.size, args.seed)
        if args.size == 1000:
            splits['test_ood'] = counterfactual_set(30, args.seed, ood=True)
            splits['counterfactual'] = counterfactual_set(30, args.seed)
    validation = validate_splits(splits)
    for split, rows in splits.items():
        write_jsonl(args.output / f'{split}.jsonl', rows)
    main_rows = [r for name, rows in splits.items() if name not in {'test_ood', 'counterfactual'} for r in rows]
    summary = {'seed': args.seed, 'source_url': SOURCE_URL,
        'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'source_license': LICENSE, 'main': statistics(main_rows),
        'splits': {s: statistics(rows) for s, rows in splits.items()}, 'validation': validation}
    (args.output / 'statistics.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'validation': validation, 'main': summary['main']}, indent=2))


if __name__ == '__main__':
    main()
