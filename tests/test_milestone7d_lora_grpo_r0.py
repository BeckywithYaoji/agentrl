from pathlib import Path

from scripts.milestone7d_lora_grpo_r0 import (
    BEST_CHECKPOINT, LATEST_CHECKPOINT, CHECKPOINT_ROOT, FORMAL_STEPS,
    SELECTION_VAL_SIZE, LOCKED_VAL_SIZE, validate_checkpoint_metadata,
)


def test_m7d_checkpoint_layout_and_steps():
    assert str(CHECKPOINT_ROOT) == '/root/autodl-tmp/checkpoints/agentrl_m7d'
    assert BEST_CHECKPOINT.name == 'best'
    assert LATEST_CHECKPOINT.name == 'latest'
    assert FORMAL_STEPS == 100
    assert SELECTION_VAL_SIZE == 20
    assert LOCKED_VAL_SIZE == 100


def test_m7d_script_locks_r0_and_lora_only_path():
    text = Path('scripts/milestone7d_lora_grpo_r0.py').read_text()
    assert 'verl_r0_reward.py' in text
    assert 'SearchXMLAgentLoop' in text
    assert 'load_lora_training_state' in text
    assert 'save_lora_training_state' in text
    assert 'optimizer_frozen_numel' in text
    assert 'q_proj' in text and 'o_proj' in text


def test_m7d_script_does_not_touch_frozen_eval_sets():
    text = Path('scripts/milestone7d_lora_grpo_r0.py').read_text()
    assert 'test.jsonl' not in text
    assert 'test_ood' not in text
    assert 'counterfactual' not in text


def test_m7d_metadata_accepts_val_selected_best_step():
    ok, message = validate_checkpoint_metadata({
        'global_step': 25,
        'trainable_params': 3_211_264,
        'optimizer_owned_params': 3_211_264,
        'optimizer_frozen_numel': 0,
    }, expected_global_step=25)
    assert ok, message


def test_m7d_val_diagnostics_match_by_question_and_dumped_gts():
    text = Path('scripts/milestone7d_lora_grpo_r0.py').read_text()
    assert "if row['question'] in record['input']" in text
    assert "record.get('gts', source['answer'])" in text


def test_m7d_sample_id_is_carried_in_reward_model_for_audit_only():
    text = Path('scripts/milestone7d_lora_grpo_r0.py').read_text()
    assert "'sample_id': sample['id']}" in text
    assert "select_fields=['reward_model']" in text
