from agentrl.verl_r0_reward import compute_score

def test_r0_extracts_final_answer():
    assert compute_score('src','<search>x</search><answer> Alice Example </answer>','Alice Example',{})==1.0
    assert compute_score('src','<search>x</search><answer>Wrong</answer>','Alice Example',{})==0.0
    assert compute_score('src','<search>x</search>','Alice Example',{})==0.0
    assert compute_score('src','<answer>Alice Example</answer><answer>Wrong</answer>','Alice Example',{})==0.0

def test_protocol_diagnostics_do_not_change_r0():
    assert compute_score('src','<answer>Alice Example</answer>','Alice Example',{})==1.0
    assert compute_score('src','reason <answer>Alice Example</answer>','Alice Example',{})==1.0
