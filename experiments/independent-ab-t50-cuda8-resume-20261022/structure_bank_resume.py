"""Exact task-boundary recovery for the independent A/B bank control only."""
import json
import logging
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.canonical_hash import hash_named_tensors, model_tensor_map
from utils.rng_utils import rng_preserving, deterministic_loader


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def restore(args, data_manager):
    from utils import factory
    assert args['sa_structure_mode'] == 'independent_bank'
    assert not args.get('distributed')
    assert args['memory_size'] == 0 and not args['sa_adaptive_a_enabled']
    bundle = Path(args['sa_resume_boundary_from'])
    metadata = json.loads((bundle / 'boundary.json').read_text())
    completed = metadata['tasks_completed']
    assert metadata['seed'] == args['seed'] == 1993
    assert metadata['class_order'] == [int(x) for x in data_manager._class_order]
    assert 0 < completed < data_manager.nb_tasks
    output = Path(args['filepath'])
    state = torch.load(bundle / 'sa_state.pt', map_location='cpu', weights_only=True)
    assert state['structure_mode'] == 'independent_bank'
    assert len(state['entries']) == completed and state['task_id'] == completed - 1
    for name in ['sa_state.pt', 'sa_merged_lora.pt', 'sa_prototypes.pt', 'sa_coordinate_diagnostics.json']:
        shutil.copy2(bundle / name, output / name)
    old_hashes = json.loads((bundle / 'p0_hashes.json').read_text())
    assert set(old_hashes) == {str(x) for x in range(completed)}
    write_json(output / 'p0_hashes.json', old_hashes)
    # Learner construction builds a temporary Task0 network. Historical bank
    # artifacts are loaded only by the explicit deployment rebuild below.
    actual_filepath = args['filepath']
    args['filepath'] = str(output / '_empty_constructor') + '/'
    try:
        model = factory.get_model(args['model_name'], args)
    finally:
        args['filepath'] = actual_filepath
    model._cur_task = completed - 1
    model._known_classes = model._total_classes = sum(data_manager._increments[:completed])
    net = model._raw_network()
    net.backbone = model.update_network(index=False, task_index=completed, initialize_training_partition=False)
    net.update_fc(model._total_classes)
    weight = torch.load(bundle / 'fc_weight.pt', map_location='cpu', weights_only=True)
    bias = torch.load(bundle / 'fc_bias.pt', map_location='cpu', weights_only=True)
    assert tuple(weight.shape) == (model._total_classes, 768)
    with torch.no_grad():
        net.fc.weight.copy_(weight)
        net.fc.bias.copy_(bias)
    prototypes = torch.load(output / 'sa_prototypes.pt', map_location='cpu', weights_only=True)
    assert set(prototypes) == set(range(model._total_classes))
    net.set_prototypes(prototypes)
    net.set_head_mode('proto')
    net.to(model._device)
    net.eval()
    actual = hash_named_tensors(model_tensor_map(net, include_eval_scalars=False))
    assert actual == metadata['model_hash'], ('RESTORED_MODEL_HASH_MISMATCH', actual, metadata['model_hash'])
    audit = dict(passed=True, restored_tasks=completed, restored_classes=model._total_classes,
                 model_hash=actual, model_bitwise_equal=True, original_boundary=str(bundle))
    if args.get('sa_resume_verify_eval', False):
        dataset = data_manager.get_dataset(np.arange(model._total_classes), source='test', mode='test')
        model._eval_test_dataset = dataset
        model.test_loader = deterministic_loader(dataset, batch_size=args['batch_size'], shuffle=False,
                                                num_workers=0, pin_memory=True,
                                                seed=int(args['seed']) + (completed - 1) * 1000 + 2)
        with rng_preserving():
            score, nme = model.eval_task()
        assert nme is None
        expected = metadata['cnn_curve']['top1'][-1]
        assert abs(float(score['top1']) - expected) < 1e-9, ('BOUNDARY_SCORE_MISMATCH', score['top1'], expected)
        grouped = score['grouped']
        row = [float(grouped[k]) for k in sorted(k for k in grouped if '-' in k)]
        assert row == metadata['cnn_matrix'][-1], ('BOUNDARY_TASK_ACCURACIES_MISMATCH', row)
        audit.update(test_top1=float(score['top1']), test_top1_matches_original=True,
                     per_task_accuracy_matches_original=True)
    write_json(output / 'restore_audit.json', audit)
    logging.info('[BoundaryResume] restored %d completed tasks, %d classes, exact hash=%s',
                 completed, model._total_classes, actual)
    return model, metadata


def checkpoint(model, data_manager, args, task, cnn_curve, nme_curve, cnn_matrix, nme_matrix):
    output = Path(args['filepath'])
    p0 = json.loads((output / 'p0_hashes.json').read_text())
    expected = args.get('sa_resume_replay_expected', {})
    if str(task) in expected:
        for key, value in expected[str(task)].items():
            assert p0[str(task)][key] == value, ('REPLAY_TRAINING_MISMATCH', task, key, p0[str(task)][key], value)
        write_json(output / 'replay_training_audit.json', dict(passed=True, task_id=task,
                   post_train_model_bitwise_equal=True, post_train_rng_bitwise_equal=True))
    target = Path(args['sa_boundary_checkpoint_root']) / f'task_{task+1:03d}'
    assert not target.exists(), ('Checkpoint already exists', str(target))
    staging = target.with_name(target.name + '.building')
    staging.mkdir(parents=True)
    for name in ['sa_state.pt', 'sa_merged_lora.pt', 'sa_prototypes.pt',
                 'sa_coordinate_diagnostics.json', 'p0_hashes.json']:
        shutil.copy2(output / name, staging / name)
    net = model._raw_network()
    torch.save(net.fc.weight.detach().cpu(), staging / 'fc_weight.pt')
    torch.save(net.fc.bias.detach().cpu(), staging / 'fc_bias.pt')
    state = torch.load(staging / 'sa_state.pt', map_location='cpu', weights_only=True)
    assert state['task_id'] == task and len(state['entries']) == task + 1
    assert len(cnn_curve['top1']) == len(cnn_matrix) == task + 1
    model_hash = hash_named_tensors(model_tensor_map(net, include_eval_scalars=False))
    assert model_hash == p0[str(task)]['post_eval_model_hash']
    metadata = dict(version=1, mode='independent_bank', tasks_completed=task+1,
                    seed=args['seed'], class_order=[int(x) for x in data_manager._class_order],
                    model_hash=model_hash, cnn_curve=cnn_curve, nme_curve=nme_curve,
                    cnn_matrix=cnn_matrix, nme_matrix=nme_matrix,
                    source_output=str(output), optimizer_resets_each_task=True)
    write_json(staging / 'boundary.json', metadata)
    numpy_state = np.random.get_state()
    torch.save(dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all(),
                    python=random.getstate(), numpy=(numpy_state[0], numpy_state[1].tolist(),
                    numpy_state[2], numpy_state[3], numpy_state[4])), staging / 'rng_state.pt')
    staging.rename(target)
    write_json(Path(args['sa_boundary_checkpoint_root']) / 'latest.json',
               dict(tasks_completed=task+1, checkpoint=str(target)))
    logging.info('[BoundaryCheckpoint] committed %s', target)
