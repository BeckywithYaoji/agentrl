import torch

from scripts.milestone7a_qwen17b_qlora_smoke import (
    has_searchxml_action,
    select_smoke_rows,
    validate_only_adapter_trainable,
)


class TinyAdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.lora_A = torch.nn.Parameter(torch.zeros(2, 2))
        for param in self.base.parameters():
            param.requires_grad = False


def test_validate_only_adapter_trainable_accepts_lora_only_optimizer():
    model = TinyAdapterModel()
    optimizer = torch.optim.AdamW([model.lora_A], lr=1e-3)
    report = validate_only_adapter_trainable(model, optimizer)
    assert report['trainable_params'] == 4
    assert report['trainable_ratio'] < 1.0


def test_validate_only_adapter_trainable_rejects_base_param():
    model = TinyAdapterModel()
    model.base.weight.requires_grad = True
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    try:
        validate_only_adapter_trainable(model, optimizer)
    except RuntimeError as exc:
        assert 'non-adapter trainable' in str(exc)
    else:
        raise AssertionError('expected non-adapter trainable parameter rejection')


def test_select_smoke_rows_mixes_search_and_direct_records():
    records = [
        {'id': 's1', 'task_type': 'lookup'},
        {'id': 's2', 'task_type': 'lookup'},
        {'id': 'd1', 'task_type': 'direct_answer'},
        {'id': 'd2', 'task_type': 'direct_answer'},
    ]
    chosen = select_smoke_rows(records, 4)
    assert [row['id'] for row in chosen] == ['s1', 's2', 'd1', 'd2']


def test_has_searchxml_action_requires_complete_tag_pair():
    assert has_searchxml_action('<search>capital of France</search>')
    assert has_searchxml_action('<answer>Paris</answer>')
    assert not has_searchxml_action('<search>capital')
