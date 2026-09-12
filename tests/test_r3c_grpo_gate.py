import pytest
from scripts.milestone5br3_grpo_smoke import require_reward_variance


def test_positive_variance_permits_update():
    assert require_reward_variance({'prompt': [0, 1, 0, 0]}, n=4)['prompt']['variance'] > 0


@pytest.mark.parametrize('groups', [
    {'prompt': [0, 0, 0, 0]},
    {'prompt': [1, 1, 1, 1]},
])
def test_zero_variance_stops_before_update(groups):
    with pytest.raises(RuntimeError, match='zero reward variance'):
        require_reward_variance(groups, n=4)


def test_incomplete_group_stops_before_update():
    with pytest.raises(ValueError):
        require_reward_variance({'prompt': [0, 1]}, n=4)


def test_missing_rollout_step_tags_use_previous_completed_actor_step():
    from scripts.milestone5br3_grpo_smoke import fill_rollout_step_tags
    tags = [{'min_global_steps': None, 'max_global_steps': None}]
    assert fill_rollout_step_tags(tags, trainer_step=2) == [{'min_global_steps': 1, 'max_global_steps': 1}]


def test_reported_rollout_step_tags_are_preserved():
    from scripts.milestone5br3_grpo_smoke import fill_rollout_step_tags
    tags = [{'min_global_steps': 0, 'max_global_steps': 0}]
    assert fill_rollout_step_tags(tags, trainer_step=2) == tags
