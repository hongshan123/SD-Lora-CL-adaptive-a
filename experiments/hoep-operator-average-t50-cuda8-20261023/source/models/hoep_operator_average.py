"""Average the arriving branch after its last quarter of optimizer updates."""
import hashlib
import json
import math
from pathlib import Path
import torch

from backbone.branch_operator_average import OperatorMean, averaging_window
from backbone.branch_operator_average import current_operators, project_to_final_a
from models.hoep_adaptation import Learner as Parent
from utils.rng_utils import rng_state_hash


class Learner(Parent):
    def __init__(self, args):
        self.average_fraction = float(args.get('sa_branch_average_fraction', .25))
        averaging_window(20, self.average_fraction)
        if args.get('sa_normalize_current_branch') or args.get('sa_live_a_absorb_mode') != 'operator_preserving_absorb':
            raise ValueError('raw current branch and exact absorption required')
        if not args.get('sa_cumulative_state') or args.get('sa_cumulative_merge') != 'live_a_aggregate_b':
            raise ValueError('fixed aggregate-G implementation required')
        if args.get('sa_total_history_weight', 0) or args.get('sa_allseen_proto_weight', 0):
            raise ValueError('isolate branch averaging from additional training losses')
        if args.get('sa_a_gain_experiment') or args.get('sa_b_metric_kind', 'off') != 'off':
            raise ValueError('original HOEP optimizer required')
        if not args.get('sa_adaptive_a_enabled') or args.get('sa_adaptive_a_strategy') != 'operator_energy_partition':
            raise ValueError('the original HOEP partition/retraction is required')
        if args.get('sa_hoep_budget_allocation', 'global') != 'global':
            raise ValueError('this initial ablation uses the original global allocation')
        if args.get('sa_gradient_accumulation_steps', 1) != 1 or args.get('distributed'):
            raise ValueError('single-GPU physical batch128 protocol required')
        super().__init__(args)
        self._operator_mean = None

    def _train(self, train_loader, test_loader):
        self._average_steps = 0
        self._operator_mean = None
        self._average_witnesses = []
        self._average_loader_batches = len(train_loader)
        self._average_training_images = len(train_loader.dataset)
        self._average_drop_last = bool(train_loader.drop_last)
        self._average_expected = len(train_loader) * int(self.args['epochs']) if self._cur_task else 0
        self._average_start, self._average_count = (
            averaging_window(self._average_expected, self.average_fraction)
            if self._cur_task else (0, 0))
        return super()._train(train_loader, test_loader)

    @staticmethod
    def _branch_tensors(backbone):
        down = torch.stack([m.weight.detach() for m in backbone.w_As])
        up = torch.stack([m.weight.detach() for m in backbone.w_Bs])
        if len(down) != 24 or len(up) != 24:
            raise ValueError('expected all twelve Q/V branch pairs')
        return down, up, backbone.wrapped_param[0].param.detach().clone()

    def _after_optimizer_step(self, inputs=None, targets=None, optimizer=None):
        result = super()._after_optimizer_step(inputs, targets, optimizer)
        if not self._cur_task:
            return result
        self._average_steps += 1
        if self._average_steps > self._average_expected:
            raise RuntimeError('more optimizer callbacks than the fixed protocol')
        if self._average_steps >= self._average_start:
            before_rng = rng_state_hash()
            a, b, scale = self._branch_tensors(self._raw_network().backbone)
            if self._operator_mean is None:
                self._operator_mean = OperatorMean()
            self._operator_mean.add(current_operators(a, b, scale))
            folder = Path(self.args['filepath']) / 'branch_average' / f'task_{self._cur_task:03d}'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'step_{self._average_steps:06d}.pt'
            if path.exists():raise FileExistsError(path)
            torch.save(dict(task_id=self._cur_task, step=self._average_steps,
                            down=a.cpu(), current_up=b.cpu(), current_scale=scale.cpu()), path)
            self._average_witnesses.append(dict(step=self._average_steps,
                file=str(path.relative_to(self.args['filepath'])),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if rng_state_hash() != before_rng:
                raise RuntimeError('averaging witnesses changed training RNG')
        return result

    @torch.no_grad()
    def _apply_average(self, raw_network):
        if self._average_steps != self._average_expected:
            raise RuntimeError('optimizer callback count does not match the original schedule')
        if self._operator_mean is None or self._operator_mean.count != self._average_count:
            raise RuntimeError('the averaging window was not actually exercised')
        before_rng = rng_state_hash()
        backbone = raw_network.backbone
        a, b, scale = self._branch_tensors(backbone)
        historical = torch.stack([getattr(block.attn.qkv, 'aggregate_' + suffix).detach()
                                  for block in backbone.lora_vit.blocks for suffix in ('q', 'v')])
        last = current_operators(a, b, scale)
        mean = self._operator_mean.mean
        coefficient, projected, diagnostics = project_to_final_a(mean, a)
        folder = Path(self.args['filepath']) / 'branch_average'
        folder.mkdir(exist_ok=True)
        archive = folder / f'task_{self._cur_task:03d}_pre_average.pt'
        if archive.exists():
            raise FileExistsError(archive)
        torch.save(dict(task_id=self._cur_task, stage='post_optimizer_pre_average',
                        down=a.detach().cpu(), current_up=b.detach().cpu(),
                        current_scale=scale.detach().cpu().clone(),
                        historical_up=historical.cpu(),
                        mean_current_operators=mean.detach().cpu()), archive)
        for module, value in zip(backbone.w_Bs, coefficient):
            module.weight.copy_(value.to(module.weight))
        # s*B is the coefficient: s=1 avoids division by a small learned scalar.
        backbone.wrapped_param[0].param.fill_(1.)
        after_a, after_b, after_scale = self._branch_tensors(backbone)
        assert torch.equal(after_a, a), 'averaging changed A'
        after_history = torch.stack([getattr(block.attn.qkv, 'aggregate_' + suffix).detach()
                                     for block in backbone.lora_vit.blocks for suffix in ('q', 'v')])
        assert torch.equal(after_history, historical), 'averaging changed historical G'
        actual = current_operators(after_a, after_b, after_scale)
        error = float(torch.linalg.vector_norm(actual - projected)
                      / torch.linalg.vector_norm(projected).clamp_min(1e-30))
        if not math.isfinite(error) or error > 5e-6:
            raise FloatingPointError('written current branch differs from the projected mean')
        if rng_state_hash() != before_rng:
            raise RuntimeError('operator averaging changed training RNG')
        diagnostics.update(
            task=self._cur_task, enabled=True, fraction=self.average_fraction,
            optimizer_steps=self._average_steps, expected_optimizer_steps=self._average_expected,
            window_start_step=self._average_start, average_count=self._operator_mean.count,
            train_loader_batches=self._average_loader_batches,
            current_training_images=self._average_training_images,
            train_loader_drop_last=self._average_drop_last,
            witnesses=self._average_witnesses,
            current_write_relative_error=error, a_unchanged=True, rng_unchanged=True,
            historical_g_unchanged=True,
            last_current_operator_frobenius=float(last.square().sum().sqrt()),
            relative_current_operator_change=float(torch.linalg.vector_norm(actual - last)
                / torch.linalg.vector_norm(last).clamp_min(1e-30)),
            original_scale=float(scale), written_scale=1.,
            accumulator_bytes=mean.numel() * mean.element_size(),
            pre_average_archive=str(archive),
            pre_average_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            deployment_extra_parameters=0)
        self._operator_mean = None
        return diagnostics

    def _before_task_save(self, raw_network, train_loader):
        record = self._apply_average(raw_network) if self._cur_task else dict(
            task=0, enabled=False, task0_unchanged=True, average_count=0,
            optimizer_steps=0, deployment_extra_parameters=0)
        super()._before_task_save(raw_network, train_loader)
        folder = Path(self.args['filepath']) / 'branch_average'
        folder.mkdir(exist_ok=True)
        path = folder / f'task_{self._cur_task:03d}.json'
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
