"""Training-only, fixed-memory LoRA controls; no historical images are replayed.

Historical second moments are accumulated once, on each task's own training
images with test preprocessing, after deployment. Moments are not refreshed
using old images. The loss acts on the TOTAL effective operator, including B.
Prototype targets are detached, initialized from current training data in the
previous deployment, and updated only AFTER computing each batch's loss.
"""
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.sa_sdlora import Learner as BaseLearner
from utils.rng_utils import deterministic_loader, rng_preserving, rng_state_hash


def response_energy(up, down, moment):
    """E||up @ down @ x||^2 using low-rank contractions, in float64."""
    up, down, moment = up.double(), down.double(), moment.double()
    return ((up.T @ up) * ((down @ moment) @ down.T)).sum()


def update_energy(up, down, old_up, old_down, moment):
    """Exact trace loss; float64 avoids cancellation near the task anchor."""
    u = torch.cat((up.double(), -old_up.double()), dim=1)
    v = torch.cat((down.double(), old_down.double()), dim=0)
    return response_energy(u, v, moment).clamp_min(0)


def lagged_prototype_loss(features, targets, prototypes, temperature):
    # Clone: the caller subsequently updates the EMA in place before backward.
    bank = F.normalize(prototypes.detach().clone().float(), dim=1)
    logits = F.normalize(features.float(), dim=1) @ bank.T / temperature
    return F.cross_entropy(logits, targets)


class Learner(BaseLearner):
    def __init__(self, args):
        self.history_weight = float(args.get('sa_total_history_weight', 0))
        self.proto_weight = float(args.get('sa_allseen_proto_weight', 0))
        self.proto_temperature = float(args.get('sa_allseen_proto_temperature', 0.05))
        self.proto_momentum = float(args.get('sa_allseen_proto_momentum', 0.1))
        for value in (self.history_weight, self.proto_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError('loss weights must be finite and nonnegative')
        if not 0 < self.proto_temperature or not 0 < self.proto_momentum <= 1:
            raise ValueError('invalid prototype temperature/momentum')
        if args.get('distributed') or len(args['device']) != 1:
            raise ValueError('this experiment implements single-GPU batch128 only')
        if args.get('sa_resume'):
            raise ValueError('resume requires restoring history/EMA state; use a fresh output')
        if args.get('sa_live_a_absorb_mode') != 'operator_preserving_absorb':
            raise ValueError('paired experiment requires exact absorption (NormCap off)')
        super().__init__(args)
        self._history_moments = []
        self._history_count = 0
        self._history_images = 0
        self._history_anchors = None
        self._proto_matrix = None
        self._proto_initial = None
        self._loss_records = {}

    def incremental_train(self, data_manager):
        self._history_anchors = None
        self._proto_matrix = None
        self._proto_initial = None
        self._loss_records = {}
        if self.proto_weight > 0 and self._cur_task >= 0:
            before = rng_state_hash()
            modes = [(m, m.training) for m in self._raw_network().modules()]
            try:
                features, labels = self._extract_current_task_features(
                    data_manager, task_index=self._cur_task + 1,
                    normalize=True, return_targets=True)
            finally:
                for module, mode in modes:
                    module.training = mode
            old = torch.load(Path(self.args['filepath']) / 'sa_prototypes.pt',
                             map_location='cpu', weights_only=True)
            assert set(old) == set(range(self._known_classes))
            end = self._known_classes + data_manager.get_task_size(self._cur_task + 1)
            bank = [F.normalize(old[c].float(), dim=0) for c in range(self._known_classes)]
            for c in range(self._known_classes, end):
                assert bool((labels == c).any())
                bank.append(F.normalize(features[labels == c].mean(0), dim=0))
            self._proto_matrix = torch.stack(bank).to(self._device)
            self._proto_initial = self._proto_matrix.detach().cpu().clone()
            if before != rng_state_hash():
                raise RuntimeError('prototype initialization changed training RNG')
        super().incremental_train(data_manager)
        # Parent has completed exact absorption, transport and formal prototypes.
        if self.history_weight > 0:
            self._collect_current_moments(data_manager)
        self._save_adaptation_state()

    def _total_history_loss(self):
        backbone = self._raw_network().backbone
        wrappers = backbone._recoverability_wrappers()
        assert len(wrappers) == len(self._history_moments) == 12
        if self._history_anchors is None:
            anchors = []
            for wrapper in wrappers:
                for branch in ('q', 'v'):
                    a = getattr(wrapper, 'a_' + branch).weight.detach().clone()
                    b = getattr(wrapper, 'b_' + branch).weight.detach()
                    assert torch.count_nonzero(b).item() == 0, 'anchor must precede first SGD step'
                    g = getattr(wrapper, 'aggregate_' + branch).detach().clone()
                    anchors.append((g / (torch.linalg.vector_norm(a) + 1e-8), a))
            self._history_anchors = anchors
            self._history_denominator = sum(
                response_energy(up, down, self._history_moments[i // 2])
                for i, (up, down) in enumerate(anchors)).detach().clamp_min(1e-12)
        terms = []
        for j, wrapper in enumerate(wrappers):
            for k, branch in enumerate(('q', 'v')):
                a = getattr(wrapper, 'a_' + branch).weight
                b = getattr(wrapper, 'b_' + branch).weight
                g = getattr(wrapper, 'aggregate_' + branch)
                scale = wrapper.scaling_cur[0].param.reshape(())
                up = g / (torch.linalg.vector_norm(a) + 1e-8) + scale * b
                old_up, old_down = self._history_anchors[2*j+k]
                terms.append(update_energy(up, a, old_up, old_down, self._history_moments[j]))
        return (sum(terms) / self._history_denominator).float()

    def _additional_training_losses(self, inputs=None, targets=None, features=None):
        losses = super()._additional_training_losses(inputs, targets, features)
        if self._cur_task == 0:
            return losses  # Common Task 0 in every arm.
        if self.history_weight > 0:
            losses['total_history'] = self.history_weight * self._total_history_loss()
        if self.proto_weight > 0:
            assert features is not None and targets is not None
            assert self._proto_matrix.shape[0] == self._total_classes
            losses['allseen_prototype'] = self.proto_weight * lagged_prototype_loss(
                features, targets, self._proto_matrix, self.proto_temperature)
            with torch.no_grad():
                normalized = F.normalize(features.detach().float(), dim=1)
                for c in targets.unique().tolist():
                    assert c >= self._known_classes
                    mean = F.normalize(normalized[targets == c].mean(0), dim=0)
                    self._proto_matrix[c] = F.normalize(
                        (1-self.proto_momentum) * self._proto_matrix[c]
                        + self.proto_momentum * mean, dim=0)
        for key, value in losses.items():
            if not bool(torch.isfinite(value)):
                raise FloatingPointError(f'nonfinite {key} at task {self._cur_task}')
            record = self._loss_records.setdefault(key, {'sum': 0., 'steps': 0, 'max': 0.})
            scalar = float(value.detach())
            record['sum'] += scalar
            record['steps'] += 1
            record['max'] = max(record['max'], scalar)
        return losses

    def _after_backward(self, inputs=None, targets=None, optimizer=None):
        super()._after_backward(inputs, targets, optimizer)
        if self._cur_task > 0 and (self.history_weight or self.proto_weight):
            b = self._raw_network().backbone
            for module in list(b.w_As) + list(b.w_Bs) + list(b.wrapped_param):
                for parameter in module.parameters():
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError('nonfinite LoRA gradient')

    def _before_task_save(self, raw_network, train_loader):
        super()._before_task_save(raw_network, train_loader)
        out = Path(self.args['filepath']) / 'adaptation_state'
        out.mkdir(exist_ok=True)
        torch.save({
            'task': self._cur_task,
            'history_input_tokens_before_task': self._history_count,
            'history_input_images_before_task': self._history_images,
            'anchors': None if self._history_anchors is None else [
                (u.detach().cpu(), a.detach().cpu()) for u, a in self._history_anchors],
            'prototype_initial': self._proto_initial,
            'prototype_final_ema': None if self._proto_matrix is None else self._proto_matrix.detach().cpu(),
            'loss_records': self._loss_records,
        }, out / f'task_{self._cur_task:03d}_training.pt')

    @torch.no_grad()
    def _collect_current_moments(self, data_manager):
        before = rng_state_hash()
        raw = self._raw_network()
        modes = [(m, m.training) for m in raw.modules()]
        handles = []
        sums = [torch.zeros(768, 768, device=self._device) for _ in range(12)]
        counts = [0] * 12
        images = 0
        def hook(index):
            def capture(module, inputs):
                x = inputs[0].detach().float().reshape(-1, 768)
                sums[index].addmm_(x.T, x)
                counts[index] += len(x)
            return capture
        try:
            raw.eval()
            for i, block in enumerate(raw.backbone.lora_vit.blocks):
                handles.append(block.attn.qkv.register_forward_pre_hook(hook(i)))
            with rng_preserving():
                dataset = data_manager.get_dataset(
                    np.arange(self._known_classes, self._total_classes), source='train', mode='test')
                loader = deterministic_loader(dataset, batch_size=64, num_workers=0, seed=0)
                for _, inputs, labels in loader:
                    assert int(labels.min()) >= self._known_classes
                    assert int(labels.max()) < self._total_classes
                    raw.backbone(inputs.to(self._device, non_blocking=True))
                    images += len(inputs)
        finally:
            for handle in handles:
                handle.remove()
            for module, mode in modes:
                module.training = mode
        assert len(set(counts)) == 1 and counts[0] > 0
        old_count = self._history_count
        new_count = old_count + counts[0]
        updated = []
        for i, current in enumerate(sums):
            current = current.double()
            if self._history_moments:
                current.add_(self._history_moments[i], alpha=old_count)
            updated.append(((current + current.T) * (0.5/new_count)).detach())
        self._history_moments = updated
        self._history_count = new_count
        self._history_images += images
        if before != rng_state_hash():
            raise RuntimeError('historical statistics changed training RNG')
        logging.info('[TotalHistory] task=%d current_images=%d cumulative_images=%d tokens=%d bytes=%d',
                     self._cur_task, images, self._history_images, self._history_count,
                     sum(x.numel()*x.element_size() for x in updated))

    def _save_adaptation_state(self):
        out = Path(self.args['filepath']) / 'adaptation_state'
        out.mkdir(exist_ok=True)
        if self.history_weight > 0:
            torch.save({'task': self._cur_task, 'input_tokens': self._history_count,
                        'images': self._history_images,
                        'moments': [x.cpu() for x in self._history_moments],
                        'source': 'each task training images, test preprocessing, task-end deployed encoder'},
                       out / f'task_{self._cur_task:03d}_moments.pt')
        report = {
            'task': self._cur_task, 'known_classes': self._known_classes,
            'total_classes': self._total_classes, 'history_weight': self.history_weight,
            'prototype_weight': self.proto_weight, 'history_images': self._history_images,
            'history_tokens': self._history_count,
            'statistics_bytes': sum(x.numel()*x.element_size() for x in self._history_moments),
            'losses': {k: dict(v, mean=v['sum']/v['steps']) for k,v in self._loss_records.items()},
        }
        (out / f'task_{self._cur_task:03d}.json').write_text(json.dumps(report, indent=2)+'\n')
