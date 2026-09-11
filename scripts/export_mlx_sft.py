"""Pure JSON conversion, one assistant target per record. Never starts training."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agentrl.sft_dataset import load_splits, validate_splits, write_jsonl

SYSTEM = ('Use search for questions requiring external evidence. Output only '
          '<search>query</search> or a concise <answer>answer</answer>. '
          'Use the relevant evidence in <information>, regardless of document position. '
          'If the provided evidence does not answer the question, output '
          '<answer>Insufficient information.</answer>. Answer simple arithmetic directly.')


def export_rows(rows):
    records = []
    for row in rows:
        context = [{'role': 'system', 'content': SYSTEM}]
        for index, message in enumerate(row['messages']):
            context.append({'role': 'user' if message['role'] == 'tool' else message['role'],
                            'content': message['content']})
            if message['role'] == 'assistant':
                records.append({'messages': [dict(m) for m in context],
                                'canonical_id': row['id'], 'target_message_index': index})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('canonical', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('use an empty export directory')
    splits = load_splits(args.canonical)
    validate_splits(splits)
    for source, target in [('train', 'train'), ('val', 'valid')]:
        records = export_rows(splits[source])
        write_jsonl(args.output / f'{target}.jsonl', records)
        print(f'{source}: {len(splits[source])} trajectories -> {len(records)} assistant targets ({target}.jsonl)')
    print('Pure export only. Future prompt masking must supervise only the last assistant target; no test sets exported.')


if __name__ == '__main__':
    main()
