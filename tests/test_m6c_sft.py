from scripts.milestone6c_sft import select_best, validation_steps


def test_best_selection_uses_teacher_forced_val_loss_only():
    assert select_best(1.0, None)
    assert select_best(0.8, 1.0)
    assert not select_best(1.0, 0.8)
    assert not select_best(0.8, 0.8)


def test_validation_steps_include_regular_intervals_and_final():
    assert validation_steps(total_steps=370, every=93) == [93, 186, 279, 370]
