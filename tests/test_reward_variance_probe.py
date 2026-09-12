import pytest
from scripts.milestone5br3_reward_variance_probe import summarize_groups


def rows(rewards):
    return [dict(event='trajectory', sample_id='fixed', rollout_index=i, R0=r,
                 final_answer=str(i), protocol_valid=True) for i, r in enumerate(rewards)]


def test_mixed_rewards_have_population_variance():
    group = summarize_groups(rows([0, 0, 1, 1]), 4)[0]
    assert group['rewards'] == [0, 0, 1, 1]
    assert group['mean'] == 0.5
    assert group['variance'] == 0.25
    assert group['std'] == 0.5
    assert len(group['rollouts']) == 4


def test_constant_rewards_have_zero_variance():
    assert summarize_groups(rows([0, 0, 0, 0]), 4)[0]['variance'] == 0


def test_incomplete_group_is_not_accepted():
    with pytest.raises(ValueError):
        summarize_groups(rows([0, 1]), 4)


def test_duplicate_rollout_index_is_not_accepted():
    records = rows([0, 1])
    records[1]['rollout_index'] = 0
    with pytest.raises(ValueError):
        summarize_groups(records, 2)
