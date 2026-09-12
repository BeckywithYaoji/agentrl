"""Independent token-origin audit for the native GPU smoke."""
import pytest
from scripts.milestone5br3_native_rollout import audit_tokens


def trace():
    return [dict(prompt_ids=[10], token_ids=[20, 21]),
            dict(prompt_ids=[10, 20, 21, 30, 31], token_ids=[22])]


def test_mask_matches_actual_generation_and_context():
    assert audit_tokens([10], [20, 21, 30, 31, 22], [1, 1, 0, 0, 1], trace()) == (3, 2)


@pytest.mark.parametrize('ids,mask', [
    ([20, 21, 30, 31, 22], [1, 1, 1, 0, 1]),
    ([20, 21, 30, 31, 22], [1, 0, 0, 0, 1]),
    ([20, 21, 30, 31, 22], [1, 1, 0, 0]),
    ([20, 21, 99, 31, 22], [1, 1, 0, 0, 1]),
])
def test_rejects_wrong_masks_or_context(ids, mask):
    with pytest.raises(ValueError):
        audit_tokens([10], ids, mask, trace())
