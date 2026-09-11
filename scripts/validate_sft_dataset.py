"""Validate canonical fields, evidence, protocol, IDs and split leakage."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agentrl.sft_dataset import load_splits, validate_splits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    splits = load_splits(args.directory)
    if not splits or not any(splits.values()):
        parser.error('no dataset rows found')
    print(json.dumps(validate_splits(splits), indent=2))
