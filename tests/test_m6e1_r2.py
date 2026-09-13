"""R2 contract for veRL multi-turn observations and reward metadata."""
import pytest
from agentrl.reward import compute_reward, parse_trajectory
from agentrl import verl_reward_adapter

SEARCH = '<search>where is Punt</search>'
ANSWER = '<answer>Punt</answer>'
NATIVE = SEARCH + '\nuser\n<information>Doc 1 says Punt.</information>\nassistant\n<think>\n\n</think>\n\n' + ANSWER


def score(text, required):
    return verl_reward_adapter.compute_score('train', text, 'Punt', {'requires_search': required})


def test_native_observation_and_role_boundaries_are_not_assistant_actions():
    parsed = parse_trajectory(NATIVE)
    assert parsed.protocol_valid
    assert parsed.actions == (('search', 'where is Punt'), ('answer', 'Punt'))


def test_observation_content_does_not_change_protocol():
    a = parse_trajectory(SEARCH + '<information>irrelevant <answer>bait</answer></information>' + ANSWER)
    b = parse_trajectory(SEARCH + '<information></information>' + ANSWER)
    assert a.protocol_valid and b.protocol_valid
    assert a.actions == b.actions


def test_direct_answer_valid_and_malformed_action_invalid():
    assert parse_trajectory(ANSWER).protocol_valid
    assert not parse_trajectory('<search>broken<answer>Punt</answer>').protocol_valid


def test_required_search_first_action_has_no_decision_penalty():
    result = score(NATIVE, True)
    assert result['reward']['decision_correct'] == 1.0
    assert result['score'] == 1.0


def test_skipped_required_search_gets_existing_decision_penalty():
    result = score(ANSWER, True)
    assert result['reward']['decision_correct'] == 0.0
    assert result['score'] == pytest.approx(0.9)


def test_direct_answer_when_not_required_has_no_decision_penalty():
    result = score(ANSWER, False)
    assert result['reward']['decision_correct'] == 1.0
    assert result['score'] == 1.0


def test_oversearch_penalty_unchanged():
    result = score('<search>a</search><search>b</search><search>c</search>' + ANSWER, True)
    assert result['score'] == pytest.approx(0.96)
    assert result['reward']['excess_searches'] == 2


def test_same_correct_answer_with_bad_protocol_loses_existing_penalty():
    valid = score(ANSWER, False)
    invalid = score('garbage ' + ANSWER, False)
    assert valid['reward']['answer_em'] == invalid['reward']['answer_em'] == 1.0
    assert invalid['score'] == pytest.approx(valid['score'] - 0.2)


def test_r2_is_bounded_by_existing_contract():
    cases = [NATIVE, ANSWER, 'garbage ' + ANSWER, '<answer>wrong</answer>',
             '<search>a</search><search>b</search><search>c</search>' + ANSWER]
    assert all(-0.1 <= score(text, required)['score'] <= 1.0
               for text in cases for required in (False, True))


def test_missing_search_label_fails_instead_of_skipping_decision_term():
    with pytest.raises(ValueError, match='requires_search'):
        verl_reward_adapter.compute_score('train', ANSWER, 'Punt', {})


def test_reward_extra_info_uses_only_train_sample_metadata():
    builder = getattr(verl_reward_adapter, 'reward_extra_info', None)
    assert callable(builder)
    row = {'id':'train-1','question':'Where?', 'answer':'Punt',
           'metadata':{'requires_search':True,'documents':[{'is_support':True}]}}
    extra = builder(row)
    assert extra['requires_search'] is True
    assert extra['question'] == 'Where?'
    assert 'answer' not in extra and 'documents' not in extra and 'is_support' not in extra


def test_native_audit_uses_forwarded_reward_extra_info_and_raw_solution():
    from scripts import milestone6e1_r2_audit as audit
    fn = getattr(audit, 'audit_native_trajectory', None)
    assert callable(fn)
    row = {'id':'train-1','question':'Where?', 'answer':'Punt',
           'metadata':{'requires_search':True}}
    trace = {'sample_id':'train-1','native_solution_str':NATIVE,
             'reward_extra_info':{'question':'Where?','requires_search':True},
             'search_count':1,'final_answer':'Punt'}
    result = fn(trace, row)
    assert result['requires_search'] is True
    assert result['protocol_valid'] is True
    assert result['first_action'] == 'search'
    assert result['r2'] == 1.0
    assert result['components']['decision_correct'] == 1.0
