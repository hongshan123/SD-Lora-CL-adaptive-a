#!/usr/bin/env python3
"""Complete only the final evaluation of a trained HOEP direction control.

Original failed attempts are immutable. A recovered copy records the original
nonzero exit separately from the successful evaluation subprocess exit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def literal(text):
    return ast.literal_eval(re.sub(r'np\.float(?:32|64)\(([^()]*)\)', r'\1', text))


def parse_recovery_history(body, completed_reference=False):
    if not re.search(r'Task 9, Epoch 20/20\s*=>', body):
        raise ValueError('final task training did not finish all twenty epochs')
    curves = re.findall(r'CNN top1 curve:\s*(\[[^\n]+\])', body)
    curve = literal(curves[-1]) if curves else []
    groups = [literal(value) for value in re.findall(r'CNN:\s*(\{[^\n]+\})', body)]
    if completed_reference:
        if len(curve) != 10 or len(groups) != 10:
            raise ValueError('completed reference lacks ten task evaluations')
        expected = curve[-1]
        curve, groups = curve[:9], groups[:9]
    else:
        expected = None
    if len(curve) != 9 or len(groups) != 9:
        raise ValueError('recovery requires nine previous task evaluations')
    return {'curve': curve, 'groups': groups, 'expected_final': expected}


def validate_failed_attempt(proof, body):
    if proof.get('phase') != 'failed' or proof.get('returncode') != 1 or (
            'ValueError: HOEP reference counts missing task 10' not in body):
        raise ValueError('attempt did not fail at the known Task10 evaluation bug')


def forgetting_from_groups(previous, final):
    keys = sorted(key for key in final if '-' in key)
    old = keys[:-1]
    if not old:
        return 0.0
    return sum(max([row[key] for row in previous if key in row] + [final[key]])
               - final[key] for key in old) / len(old)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(entry, completed_reference=False):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from models.sa_sdlora import Learner
    from backbone.linears import PrototypeCosineHead
    from utils.data_manager import DataManager

    source = entry['recovery_source']
    source_dir = Path(source['output'])
    body = Path(source['log']).read_text(errors='replace')
    if not completed_reference:
        validate_failed_attempt(json.loads(Path(source['proof']).read_text()), body)
    history = parse_recovery_history(body, completed_reference)
    original_config = json.loads(Path(source['config']).read_text())
    config = json.loads(Path(entry['config']).read_text())
    allowed = {'prefix', 'filepath', 'sa_hoep_reference_counts_path'}
    if {k: v for k, v in config.items() if k not in allowed} != {
            k: v for k, v in original_config.items() if k not in allowed}:
        raise ValueError('recovery configuration changed scientific settings')
    if config['sa_coordinate_stable_transport'] or config['sa_live_a_absorb_mode'] != 'operator_preserving_absorb':
        raise ValueError('recovery supports the exact-absorption, no-transport protocol')
    source_state_hash = sha(source_dir / 'sa_state.pt')
    state = torch.load(source_dir / 'sa_state.pt', map_location='cpu', weights_only=True)
    if state['task_id'] != 10 or state['rank'] != 10:
        raise ValueError('saved state does not contain all ten absorbed tasks')
    old = torch.load(source_dir / 'sa_prototypes.pt', map_location='cpu', weights_only=True)
    expected_classes = set(range(100 if completed_reference else 90))
    if set(old) != expected_classes:
        raise ValueError('saved prototypes do not match the recoverable boundary')
    target = Path(entry['output'])
    if target.exists() or Path(entry['log']).exists():
        raise FileExistsError('refusing to overwrite a recovery attempt')
    shutil.copytree(source_dir, target)
    with Path(entry['log']).open('x') as stream:
        stream.write(body + '\n[Recovery] preserved failed training log; final evaluation starts here\n')
    config['seed'] = int(config['seed'][0])
    config['device'] = [torch.device('cuda:0')]
    config.update(distributed=False, rank=0, world_size=1, local_rank=0)
    # Use the production construction and prototype routines directly. There
    # is no optimizer, training loader or call to incremental_train here.
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    manager = DataManager(config['dataset'], config['shuffle'], config['seed'],
                          config['init_cls'], config['increment'], config)
    learner = Learner.__new__(Learner)
    learner.args = config
    learner._cur_task, learner._known_classes, learner._total_classes = 9, 90, 100
    learner._device, learner.topk = config['device'][0], 5
    backbone = learner.update_network(
        index=False, task_index=10, initialize_training_partition=False,
    ).to(learner._device).eval()
    network = torch.nn.Module()
    network.backbone = backbone
    prototypes = learner._compute_prototypes(manager, network)
    if set(prototypes) != set(range(100)) or not all(
            torch.equal(prototypes[k].cpu(), old[k]) for k in range(90)):
        raise ValueError('recovery altered old prototypes or failed to construct all classes')
    reference_prototype_error = None
    if completed_reference:
        reference_prototype_error = max(
            float((prototypes[k].cpu() - old[k]).abs().max()) for k in range(90, 100))
        if reference_prototype_error > 1e-5:
            raise ValueError('recovered new prototypes differ from completed reference')
    if sha(target / 'sa_state.pt') != source_state_hash or sha(source_dir / 'sa_state.pt') != source_state_hash:
        raise ValueError('recovery altered saved A/G state')
    head = PrototypeCosineHead(prototypes).to(learner._device).eval()
    dataset = manager.get_dataset(np.arange(100), source='test', mode='test')
    loader = DataLoader(dataset, batch_size=int(config['batch_size']), shuffle=False,
                        num_workers=learner._loader_workers(), pin_memory=True)
    logits, targets, sample_ids = [], [], []
    with torch.no_grad():
        for indices, images, labels in loader:
            logits.append(head(backbone(images.to(learner._device)))['logits'].cpu())
            targets.append(labels)
            sample_ids.append(indices)
    logits, targets, sample_ids = torch.cat(logits), torch.cat(targets), torch.cat(sample_ids)
    if not torch.isfinite(logits).all():
        raise ValueError('recovered logits contain nonfinite values')
    metrics = learner._evaluate(logits.topk(5, dim=1).indices.numpy(), targets.numpy())
    if history['expected_final'] is not None and metrics['top1'] != history['expected_final']:
        raise ValueError('recovered evaluation does not reproduce completed reference accuracy')
    curve = history['curve'] + [float(metrics['top1'])]
    forgetting = forgetting_from_groups(history['groups'], metrics['grouped'])
    artifact = target / 'recovered_final_logits.pt'
    torch.save({'logits': logits, 'targets': targets, 'sample_ids': sample_ids}, artifact)
    audit = {
        'source': source, 'source_state_sha256': source_state_hash,
        'recovered_state_sha256': sha(target / 'sa_state.pt'),
        'original_training_returncode': 0 if completed_reference else 1,
        'completion_mode': 'final_evaluation_recovery', 'optimizer_steps': 0,
        'old_prototypes_bitwise_equal': True,
        'new_prototype_source': 'task9 training classes, test preprocessing, saved deployment backbone',
        'test_sample_count': len(targets), 'class_order': list(map(int, manager._class_order)),
        'fixed_backbone_source_sha256': sha(ROOT / 'backbone/sa_lora.py'),
        'final': metrics['top1'], 'aaa': sum(curve)/10, 'curve': curve,
        'forgetting': forgetting, 'reference_reproduced': completed_reference,
        'reference_new_prototype_max_abs_error': reference_prototype_error,
        'logits_file': str(artifact),
    }
    from scripts.hoep_cifar_ablation import atomic_json
    atomic_json(target / 'final_evaluation_recovery.json', audit)
    with Path(entry['log']).open('a') as stream:
        stream.write(f"\nCNN: {metrics['grouped']}\nCNN top1 curve: {curve}\n")
        stream.write(f"Average Accuracy (CNN): {audit['aaa']}\nForgetting (CNN): {forgetting}\n")
        stream.write('[Recovery] final evaluation completed with zero optimizer steps\n')
    return audit


def worker(runtime, name, pair):
    from scripts.hoep_cifar_ablation import atomic_json, verify_sources
    from scripts.hoep_cifar_parallel_queue import process_info, validated_result
    runtime = Path(runtime).resolve()
    entry = next(e for e in json.loads((runtime / 'manifest.json').read_text())['jobs'] if e['name'] == name)
    proof_path = runtime / 'parallel/job_results' / (name + '.json')
    proof = {'name': name, 'phase': 'running', 'gpu_pair': list(pair),
             'worker_pid': os.getpid(), 'worker_start_ticks': process_info(os.getpid())['start_ticks'],
             'proof_path': str(proof_path),
             'completion_mode': 'final_evaluation_recovery', 'original_training_returncode': 1}
    try:
        verify_sources(ROOT, runtime)
        atomic_json(proof_path, proof)
        source = entry['recovery_source']
        while source.get('wait_pid'):
            info = process_info(source['wait_pid'])
            if info is None or info['start_ticks'] != source['wait_start_ticks'] or info['state'] == 'Z':
                break
            time.sleep(3)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(pair[0])
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['HF_HUB_OFFLINE'] = '1'
        audit = recover(entry)
        proof.update(phase='complete', returncode=0, result=audit)
        proof['result'] = validated_result(entry, proof)
        atomic_json(proof_path, proof)
        return 0
    except Exception as error:
        proof.update(phase='failed', returncode=1, error=str(error))
        atomic_json(proof_path, proof)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-dir', required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--gpu-pair', required=True)
    parser.add_argument('--start-fd', type=int)
    args = parser.parse_args()
    if args.start_fd is not None:
        token = os.read(args.start_fd, 1)
        os.close(args.start_fd)
        if token != b'1':
            raise SystemExit('coordinator exited before recovery ownership was saved')
    pair = tuple(map(int, args.gpu_pair.split(',')))
    from scripts.hoep_cifar_stable_queue import PAIRS
    if pair not in PAIRS:
        parser.error('GPU pair uses an excluded or unauthorized card')
    raise SystemExit(worker(args.runtime_dir, args.worker, pair))
