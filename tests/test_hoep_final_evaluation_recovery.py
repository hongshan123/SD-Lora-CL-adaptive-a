"""Only the fully trained final boundary may be recovered without training."""
import importlib
import json

import pytest


def module():
    return importlib.import_module('scripts.recover_hoep_final_evaluation')


def log_fixture():
    return '\n'.join(
        f"CNN: {{'total': {90-t}, '00-09': {90-t}}}\nCNN top1 curve: {list(range(90,89-t,-1))}"
        for t in range(9)) + '\nTask 9, Epoch 20/20 => Loss 0.1\nValueError: HOEP reference counts missing task 10\n'


def test_recovery_requires_all_training_epochs_and_nine_evaluated_tasks():
    history = module().parse_recovery_history(log_fixture())
    assert history['curve'] == [90,89,88,87,86,85,84,83,82]
    with pytest.raises(ValueError, match='final task'):
        module().parse_recovery_history(log_fixture().replace('Task 9, Epoch 20/20', 'Task 9, Epoch 19/20'))
    with pytest.raises(ValueError, match='nine'):
        module().parse_recovery_history('Task 9, Epoch 20/20 => Loss 0.1\nCNN top1 curve: [90]\n')


def test_recovery_rejects_unrelated_training_failures():
    with pytest.raises(ValueError, match='Task10'):
        module().validate_failed_attempt({'phase':'failed','returncode':1}, 'CUDA out of memory')
    module().validate_failed_attempt({'phase':'failed','returncode':1}, log_fixture())


def test_forgetting_uses_previous_task_group_maxima():
    # task0 accuracy 90, task1 old-class accuracy 80. The only old task lost 10.
    assert module().forgetting_from_groups([{'00-09':90}], {'00-09':80,'10-19':95}) == 10
