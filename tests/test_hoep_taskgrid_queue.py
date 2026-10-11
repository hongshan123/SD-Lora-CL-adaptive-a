"""Completion must cover the configured task horizon and every transport boundary."""
import json

import pytest

from scripts.hoep_cifar_parallel_queue import ready_jobs, validated_result


def cell(tmp_path, tasks, transport=True):
    output = tmp_path / 'output'
    output.mkdir()
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(max_tasks=tasks, init_cls=4, increment=4,
                                      sa_coordinate_stable_transport=transport)))
    log = tmp_path / 'run.log'
    log.write_text(f'CNN top1 curve: {[90.0] * tasks}\nForgetting (CNN): 1\n')
    schedule = {'version': 1, 'seed': 1,
                'tasks': {str(t): [3] * 24 for t in range(1, tasks)}}
    (output / 'hoep_partition_schedule.json').write_text(json.dumps(schedule))
    diagnostics = [dict(task_id=t, num_prototypes=4*t, enabled=False)
                   for t in range(1, tasks)]
    (output / 'sa_coordinate_diagnostics.json').write_text(json.dumps(diagnostics))
    return dict(name='cell', method='hoep', seed=1, config=str(config),
                output=str(output), log=str(log)), dict(phase='complete', returncode=0)


@pytest.mark.parametrize('tasks', [5, 10, 20, 50])
def test_complete_variable_horizon_is_accepted(tmp_path, tasks):
    entry, proof = cell(tmp_path, tasks)
    assert len(validated_result(entry, proof)['curve']) == tasks


@pytest.mark.parametrize('tasks', [5, 20, 50])
def test_last_partition_boundary_is_required(tmp_path, tasks):
    entry, proof = cell(tmp_path, tasks)
    path = tmp_path / 'output/hoep_partition_schedule.json'
    data = json.loads(path.read_text())
    data['tasks'].pop(str(tasks-1))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=f'task {tasks-1}'):
        validated_result(entry, proof)


def test_transport_cannot_be_silently_missing(tmp_path):
    entry, proof = cell(tmp_path, 10)
    (tmp_path / 'output/sa_coordinate_diagnostics.json').unlink()
    with pytest.raises(ValueError, match='transport'):
        validated_result(entry, proof)


def test_transport_prototype_count_must_match_old_class_set(tmp_path):
    entry, proof = cell(tmp_path, 10)
    path = tmp_path / 'output/sa_coordinate_diagnostics.json'
    records = json.loads(path.read_text())
    records[-1]['num_prototypes'] = 200
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match='prototype'):
        validated_result(entry, proof)


def test_explicit_priority_can_start_one_cell_per_dataset():
    jobs = [dict(name='cifar', method='hoep', seed=1993, dispatch_priority=0),
            dict(name='inr', method='hoep', seed=1995, dispatch_priority=1),
            dict(name='cub', method='hoep', seed=1, dispatch_priority=2)]
    assert [j['name'] for j in ready_jobs(jobs, set(), set())] == ['cifar', 'inr', 'cub']
