"""Rollout-only veRL native R2 audit on two fixed train-safe prompts."""
import asyncio
import json
from pathlib import Path

from agentrl.reward import parse_trajectory
from agentrl.verl_reward_adapter import compute_score, reward_extra_info


def audit_native_trajectory(trace, sample):
    extra = trace['reward_extra_info']
    expected = reward_extra_info(sample)
    if extra != expected or trace['sample_id'] != sample['id']:
        raise ValueError('native rollout lost or changed reward extra_info')
    solution = trace['native_solution_str']
    parsed = parse_trajectory(solution)
    result = compute_score('agentrl_train', solution, sample['answer'], extra)['reward']
    if parsed.search_count != trace['search_count']:
        raise ValueError('native action and retrieval counts differ')
    correct = result['answer_em']
    protocol_penalty = (-0.2 if not parsed.protocol_valid else 0.0) if correct else (-0.1 if not parsed.protocol_valid else 0.0)
    decision_penalty = -0.1 if correct and result['decision_correct'] == 0 else 0.0
    oversearch_penalty = -min(0.1, 0.02 * result['excess_searches']) if correct else 0.0
    components = {'answer_credit':correct, 'protocol_penalty':protocol_penalty,
                  'decision_penalty':decision_penalty, 'oversearch_penalty':oversearch_penalty,
                  'decision_correct':result['decision_correct'],
                  'excess_searches':result['excess_searches']}
    if abs(sum(components[k] for k in ('answer_credit','protocol_penalty','decision_penalty','oversearch_penalty')) - result['reward']['r2']) > 1e-9:
        raise ValueError('R2 component breakdown does not sum to reward')
    return {'sample_id':sample['id'], 'requires_search':extra['requires_search'],
            'first_action':parsed.first_action, 'final_answer':parsed.answer,
            'answer_correct':bool(correct), 'protocol_valid':parsed.protocol_valid,
            'search_count':parsed.search_count, 'components':components,
            'r2':result['reward']['r2'], 'native_solution_str':solution}


def main():
    from scripts.milestone5br3_native_rollout import smoke
    from scripts.milestone5br3_concise_answer_probe import CONCISE_REMINDER
    artifact = Path('artifacts/milestone6e1/native_audit')
    artifact.mkdir(parents=True, exist_ok=True)
    asyncio.run(smoke(artifact, candidate_count=2, group_size=1, temperature=0.0,
                      answer_reminder=CONCISE_REMINDER,
                      model_path='/root/autodl-tmp/checkpoints/agentrl_m6c/best'))
    with Path('data/sft/train.jsonl').open() as handle:
        samples = [json.loads(next(handle)) for _ in range(2)]
    by_id = {sample['id']:sample for sample in samples}
    trajectories = [json.loads(line) for line in (artifact / 'trace.jsonl').open()
                    if '"event": "trajectory"' in line]
    reports = [audit_native_trajectory(t, by_id[t['sample_id']]) for t in trajectories]
    if len(reports) != 2 or not any(r['search_count'] and r['protocol_valid'] for r in reports):
        raise RuntimeError('native search trajectory did not validate under R2')
    (artifact / 'r2_audit.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    print(json.dumps({'event':'r2_audit','trajectories':reports}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
