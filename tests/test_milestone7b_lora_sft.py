import json

from scripts.milestone7b_lora_sft import CONFIG_PATH, load_config, select_best, validation_steps


def test_m7b_config_locks_model_lora_and_data_boundary():
    config = load_config()
    assert config['model'] == 'Qwen/Qwen3-1.7B'
    assert config['peft_method'] == 'bf16-lora'
    assert config['lora_target_modules'] == ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    assert config['train_path'] == 'data/sft/train.jsonl'
    assert config['val_path'] == 'data/sft/val.jsonl'
    assert config['frozen_paths_touched'] is False
    text = CONFIG_PATH.read_text()
    assert 'test.jsonl' not in text
    assert 'counterfactual.jsonl' not in text


def test_m7b_selection_rule_is_lowest_finite_val_loss():
    assert select_best(0.9, None)
    assert select_best(0.8, 0.9)
    assert not select_best(1.0, 0.9)
    assert not select_best(float('nan'), 0.9)


def test_m7b_validation_steps_include_each_epoch_and_final():
    assert validation_steps(93, 2, 1) == [93, 186]
    assert validation_steps(93, 2, 0.5) == [46, 92, 138, 184, 186]


def test_m7b_formal_config_json_is_valid():
    config = json.loads(CONFIG_PATH.read_text())
    assert config['effective_batch_size'] == config['batch_size'] * config['gradient_accumulation_steps']
    assert config['selection_rule'].startswith('lowest teacher-forced val loss')
