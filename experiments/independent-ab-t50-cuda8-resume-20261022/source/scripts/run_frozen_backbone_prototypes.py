"""Frozen pretrained ViT control with HOEP's exact prototype representation.

Only current-task training images construct new prototypes. Previous prototypes
are retained unchanged. Cached test features are used solely for evaluation;
the fixed backbone makes this equivalent to extracting them at every boundary.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import timm
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backbone.linears import PrototypeCosineHead
from trainer import _set_random, _set_deterministic_backend
from utils.canonical_hash import hash_named_tensors
from utils.data_manager import DataManager
from utils.rng_utils import deterministic_loader


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prototypes_from_features(features, labels, class_ids):
    normalized = F.normalize(features.float(), p=2, dim=1)
    result = {}
    for c in class_ids:
        selected = normalized[labels == c]
        if len(selected) == 0:
            raise ValueError('Missing training class ' + str(c))
        result[int(c)] = F.normalize(selected.mean(dim=0), p=2, dim=0)
    return result


@torch.no_grad()
def extract(model, dataset, device):
    loader = deterministic_loader(dataset, batch_size=64, shuffle=False,
                                  num_workers=2, seed=0)
    features, labels = [], []
    for _, inputs, targets in loader:
        features.append(model(inputs.to(device)).cpu())
        labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)


def run(config_path, max_tasks=None):
    config = json.loads(Path(config_path).read_text())
    output = Path(config['filepath'])
    output.mkdir(parents=True, exist_ok=False)
    seed = int(config['seed'][0])
    _set_random(seed, config)
    _set_deterministic_backend(config)
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    manager = DataManager(config['dataset'], config['shuffle'], seed,
                          config['init_cls'], config['increment'], config)
    tasks = manager.nb_tasks if max_tasks is None else max_tasks
    if not 1 <= tasks <= manager.nb_tasks:
        raise ValueError('Invalid number of tasks')
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    model.requires_grad_(False).eval().to(device)
    before_hash = hash_named_tensors(model.state_dict())
    assert not any(p.requires_grad for p in model.parameters())
    provenance = dict(dataset=config['dataset'], seed=seed, tasks=tasks,
        class_order=manager._class_order, backbone='vit_base_patch16_224',
        pretrained_config=model.default_cfg, backbone_initial_hash=before_hash,
        train_preprocessing='test', train_transform=repr(manager._test_trsf),
        prototype_rule='normalize(mean(normalize(each feature)))',
        classifier='backbone.linears.PrototypeCosineHead', trainable_parameters=0,
        gradient_updates=0, prototype_source='current task training split only',
        transport='identity: backbone fixed', normcap=False,
        cache='test features only; never used for prototype construction')
    write(output / 'provenance.json', provenance)
    prototypes, test_features, test_labels, curve, matrix = {}, [], [], [], []
    known = 0
    for task in range(tasks):
        total = known + manager.get_task_size(task)
        classes = np.arange(known, total)
        train = manager.get_dataset(classes, source='train', mode='test')
        features, labels = extract(model, train, device)
        old_hash = hash_named_tensors({str(c): v for c, v in prototypes.items()})
        new = prototypes_from_features(features, labels, classes)
        prototypes.update(new)
        assert old_hash == hash_named_tensors({str(c): prototypes[c] for c in range(known)})
        train_counts = {int(c): int((labels == c).sum()) for c in classes}
        del features, labels, train
        test = manager.get_dataset(classes, source='test', mode='test')
        features, labels = extract(model, test, device)
        test_features.append(features)
        test_labels.append(labels)
        head = PrototypeCosineHead(prototypes).to(device).eval()
        with torch.no_grad():
            all_features = torch.cat(test_features)
            all_labels = torch.cat(test_labels)
            logits = torch.cat([head(batch.to(device))['logits'].cpu()
                                for batch in all_features.split(64)])
            # Verify the cached path against a fresh model forward, independently
            # of the feature cache, at both ends of the sequence.
            if task in (0, tasks - 1):
                loader = deterministic_loader(test, batch_size=64, shuffle=False,
                                              num_workers=0, seed=0)
                _, inputs, _ = next(iter(loader))
                fresh = head(model(inputs.to(device)))['logits'].cpu()
                cached = logits[-len(labels):][:len(fresh)]
                torch.testing.assert_close(fresh, cached, atol=1e-5, rtol=1e-5)
        prediction = logits.argmax(1)
        accuracy = 100 * (prediction == all_labels).double().mean().item()
        curve.append(accuracy)
        row = []
        for old_task in range(task + 1):
            low = sum(manager._increments[:old_task])
            high = low + manager._increments[old_task]
            mask = (all_labels >= low) & (all_labels < high)
            row.append(100 * (prediction[mask] == all_labels[mask]).double().mean().item())
        matrix.append(row)
        torch.save(dict(labels=all_labels, logits=logits,
                        order='class-incremental remapped labels; dataset order within each class'),
                   output / f'task_{task:03d}_logits.pt')
        torch.save(prototypes, output / 'sa_prototypes.pt')
        write(output / f'task_{task:03d}.json', dict(task=task, total_classes=total,
              train_counts=train_counts, prototype_source='current task train, test preprocessing',
              retained_old_prototypes_unchanged=True, top1=accuracy))
        metrics = dict(complete=False, tasks_completed=task+1, tasks=tasks,
                       curve=curve, accuracy_curve=curve,
                       final_accuracy=accuracy if task+1 == tasks else None,
                       average_incremental_accuracy=sum(curve)/len(curve),
                       task_accuracy_matrix=matrix)
        write(output / 'metrics.json', metrics)
        print(f'Task {task+1}/{tasks}: seen={total}, Top1={accuracy:.4f}, AAA={np.mean(curve):.4f}', flush=True)
        known = total
    after_hash = hash_named_tensors(model.state_dict())
    if after_hash != before_hash:
        raise RuntimeError('Frozen backbone changed')
    forgetting = np.mean([max(matrix[t][j] for t in range(j, tasks)) - matrix[-1][j]
                          for j in range(tasks-1)]) if tasks > 1 else 0.0
    metrics.update(complete=True, final_forgetting=float(forgetting),
                   backbone_unchanged=True, backbone_final_hash=after_hash)
    write(output / 'metrics.json', metrics)
    print('FROZEN_BASELINE_COMPLETE ' + json.dumps(metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max-tasks', type=int)
    args = parser.parse_args()
    run(args.config, args.max_tasks)
