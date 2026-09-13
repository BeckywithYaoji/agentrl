import json
from pathlib import Path

from scripts.milestone7c_grpo_r0_smoke import validate_lora_only_audit
from scripts.milestone7c3_grpo_stability_resume import (
    CHECKPOINT,
    SAMPLE_ID,
    validate_checkpoint_metadata,
    write_checkpoint_metadata,
    read_checkpoint_metadata,
)


def test_m7c3_uses_m7b_adapter_scoped_checkpoint():
    assert str(CHECKPOINT) == '/root/autodl-tmp/checkpoints/agentrl_m7c3/resume_ckpt'
    assert SAMPLE_ID == 'squad-5731c7ade17f3d14004223d9'


def test_m7c3_checkpoint_metadata_roundtrip(tmp_path):
    metadata = {
        'global_step': 5,
        'trainable_params': 3_211_264,
        'optimizer_owned_params': 3_211_264,
        'optimizer_frozen_numel': 0,
    }
    path = write_checkpoint_metadata(tmp_path, metadata)
    assert path.name == 'agentrl_m7c3_metadata.json'
    assert read_checkpoint_metadata(tmp_path) == metadata


def test_m7c3_resume_global_step_metadata_validation():
    ok, message = validate_checkpoint_metadata({
        'global_step': 5,
        'trainable_params': 3_211_264,
        'optimizer_owned_params': 3_211_264,
        'optimizer_frozen_numel': 0,
    }, expected_global_step=5)
    assert ok
    assert 'restored' in message


def test_m7c3_resume_metadata_rejects_wrong_global_step():
    ok, message = validate_checkpoint_metadata({
        'global_step': 4,
        'trainable_params': 3_211_264,
        'optimizer_owned_params': 3_211_264,
        'optimizer_frozen_numel': 0,
    }, expected_global_step=5)
    assert not ok
    assert 'global_step' in message


def test_m7c3_lora_only_optimizer_invariant_rejects_frozen_params():
    audit = {
        'bad_trainable': [],
        'trainable_numel': 3_211_264,
        'optimizer_numel': 3_211_264 + 10,
        'optimizer_frozen_numel': 10,
    }
    ok, message = validate_lora_only_audit(audit)
    assert not ok
    assert 'optimizer' in message


def test_m7c3_script_does_not_reference_frozen_eval_paths():
    text = Path('scripts/milestone7c3_grpo_stability_resume.py').read_text()
    assert 'test.jsonl' not in text
    assert 'test_ood' not in text
    assert 'counterfactual' not in text
    assert 'SearchXMLAgentLoop' in text
    assert 'verl_r0_reward.py' in text


def test_supplemental_lora_training_state_is_saved_and_loaded():
    script = Path('scripts/milestone7c3_grpo_stability_resume.py').read_text(encoding='utf-8')
    assert 'def save_lora_training_state' in script
    assert 'def load_lora_training_state' in script
    assert 'lora_training_state_rank_' in script
    assert 'lora_optimizer_rank_' in script
    assert 'lora_extra_rank_' in script
    assert 'load_lora_training_state(str(resume_checkpoint))' in script
