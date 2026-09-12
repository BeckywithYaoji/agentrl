
import pytest

from scripts.milestone5br3_grpo_smoke import validate_checkpoint_fingerprints


def test_checkpoint_fingerprints_require_training_change_and_exact_reload():
    pre = {'parameters': {'embed': {'slice_sha256': 'a', 'l2_norm': 1.0}}}
    post = {'parameters': {'embed': {'slice_sha256': 'b', 'l2_norm': 1.1}}}
    reload = {'parameters': {'embed': {'slice_sha256': 'b', 'l2_norm': 1.1}}}
    assert validate_checkpoint_fingerprints(pre, post, reload) == ['embed']


@pytest.mark.parametrize('reload_hash', ['a', 'c'])
def test_checkpoint_fingerprints_reject_wrong_reload(reload_hash):
    pre = {'parameters': {'embed': {'slice_sha256': 'a'}}}
    post = {'parameters': {'embed': {'slice_sha256': 'b'}}}
    reload = {'parameters': {'embed': {'slice_sha256': reload_hash}}}
    with pytest.raises(RuntimeError):
        validate_checkpoint_fingerprints(pre, post, reload)
