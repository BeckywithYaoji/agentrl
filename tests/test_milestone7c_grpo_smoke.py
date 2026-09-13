from pathlib import Path

from scripts.milestone7c_grpo_r0_smoke import ADAPTER, SAMPLE_ID
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
