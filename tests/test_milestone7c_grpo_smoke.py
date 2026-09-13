from pathlib import Path

from scripts.milestone7c_grpo_r0_smoke import ADAPTER, SAMPLE_ID, validate_lora_only_audit
from scripts.milestone7c_merged_grpo_r0_smoke import MERGED_START, M7B_ADAPTER


def test_m7c_starts_from_m7b_best_adapter_for_native_audit():
    assert str(ADAPTER) == '/root/autodl-tmp/checkpoints/agentrl_m7b/best_adapter'


def test_m7c_path_b_merged_start_location_is_stage_scoped():
    assert str(MERGED_START) == '/root/autodl-tmp/checkpoints/agentrl_m7c/merged_sft_start'
    assert str(M7B_ADAPTER) == '/root/autodl-tmp/checkpoints/agentrl_m7b/best_adapter'


def test_m7c_uses_locked_train_safe_sample():
    assert SAMPLE_ID == 'squad-5731c7ade17f3d14004223d9'


def test_m7c_scripts_do_not_reference_frozen_eval_paths():
    for script in ['scripts/milestone7c_grpo_r0_smoke.py', 'scripts/milestone7c_merged_grpo_r0_smoke.py']:
        text = Path(script).read_text()
        assert 'test.jsonl' not in text
        assert 'test_ood' not in text
        assert 'counterfactual' not in text
        assert 'verl_r0_reward.py' in text
        assert 'SearchXMLAgentLoop' in text


def test_m7c2_lora_only_audit_accepts_expected_adapter_params():
    audit = {'bad_trainable': [], 'trainable_numel': 3_211_264, 'optimizer_numel': 3_211_264}
    ok, message = validate_lora_only_audit(audit)
    assert ok
    assert 'passed' in message


def test_m7c2_lora_only_audit_rejects_base_trainable_params():
    audit = {
        'bad_trainable': ['base_model.model.embed_tokens.weight'],
        'trainable_numel': 3_211_264,
        'optimizer_numel': 3_211_264,
    }
    ok, message = validate_lora_only_audit(audit)
    assert not ok
    assert 'non-LoRA' in message


def test_m7c2_lora_only_audit_rejects_optimizer_mismatch():
    audit = {'bad_trainable': [], 'trainable_numel': 3_211_264, 'optimizer_numel': 42}
    ok, message = validate_lora_only_audit(audit)
    assert not ok
    assert 'optimizer' in message


def test_m7c2_lora_only_audit_rejects_unexpected_trainable_count():
    audit = {'bad_trainable': [], 'trainable_numel': 100, 'optimizer_numel': 100}
    ok, message = validate_lora_only_audit(audit)
    assert not ok
    assert 'unexpected' in message
