"""Fixed four-prompt, four-rollout R0 variance probe using the B2 native pipeline."""
import asyncio
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pvariance


def summarize_groups(records, group_size):
    grouped = defaultdict(list)
    for record in records:
        if record['event'] == 'trajectory':
            grouped[record['sample_id']].append(record)
    result = []
    for sample_id, rollouts in grouped.items():
        rollouts.sort(key=lambda row: row['rollout_index'])
        if [r['rollout_index'] for r in rollouts] != list(range(group_size)):
            raise ValueError(f'Incomplete or duplicate rollout group: {sample_id}')
        rewards = [r['R0'] for r in rollouts]
        variance = pvariance(rewards)
        result.append(dict(sample_id=sample_id, rewards=rewards, mean=mean(rewards),
                           variance=variance, std=math.sqrt(variance), rollouts=rollouts))
    return result


def main(*, answer_reminder='', artifact_root='artifacts/milestone5br3b3'):
    from scripts.milestone5br3_native_rollout import smoke
    directory = Path(artifact_root) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    directory.mkdir(parents=True)
    try:
        asyncio.run(smoke(directory, candidate_count=4, group_size=4, temperature=1.0, top_p=1.0,
                          answer_reminder=answer_reminder))
        records = [json.loads(line) for line in (directory / 'trace.jsonl').read_text().splitlines()]
        groups = summarize_groups(records, 4)
        fixed = next(r for r in records if r['event'] == 'fixed_candidates')
        if [g['sample_id'] for g in groups] != fixed['sample_ids']:
            raise ValueError('Reported groups differ from frozen candidates')
        passed = any(g['variance'] > 0 for g in groups)
        report = dict(decision='PASS' if passed else 'BLOCKED', groups=groups,
                      sampling=fixed, gpu_peak_memory_mib=max(
                          r['gpu_peak_memory_mib'] for r in records if r['event'] == 'memory'))
        (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if passed else 1
    except Exception:
        import traceback
        (directory / 'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    raise SystemExit(main())
