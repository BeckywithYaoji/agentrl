"""Report descriptive statistics without invoking any model or trainer."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agentrl.sft_dataset import load_splits, statistics, validate_splits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    splits = load_splits(args.directory)
    if not splits or not any(splits.values()):
        parser.error('no dataset rows found')
    print(json.dumps({'validation': validate_splits(splits),
        'splits': {key: statistics(rows) for key, rows in splits.items()}}, indent=2, sort_keys=True))
