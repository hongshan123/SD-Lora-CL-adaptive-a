"""A step-size ablations inside the existing HOEP plastic directions."""
import json
import math
from pathlib import Path
import torch

from backbone.a_factor_gain import amplify_projected_gradient
from models.hoep_adaptation import Learner as Parent


class Learner(Parent):
    def __init__(self, args):
        self.a_multiplier = float(args.get('sa_a_gradient_multiplier', 1.))
        if self.a_multiplier <= 0 or not math.isfinite(self.a_multiplier):
            raise ValueError('positive finite A multiplier required')
        if args.get('optimizer') != 'sgd':
            raise ValueError('this factor-step ablation is defined for the original SGD')
        if args.get('sa_total_history_weight', 0) or args.get('sa_allseen_proto_weight', 0):
            raise ValueError('factor-step ablations isolate A gain from additional losses')
        if not args.get('sa_adaptive_a_enabled') or args.get('sa_adaptive_a_strategy') != 'operator_energy_partition':
            raise ValueError('A gain requires the existing HOEP projection/retraction')
        super().__init__(args)
        self._a_gain_stats = {}

    def incremental_train(self, data_manager):
        self._a_gain_stats = dict(steps=0, nonzero_scaled_branches=0,
                                 projected_gradient_norm_before=0., projected_gradient_norm_after=0.,
                                 max_relative_direction_error=0.)
        return super().incremental_train(data_manager)

    def _after_backward(self, inputs=None, targets=None, optimizer=None):
        super()._after_backward(inputs, targets, optimizer)
        if self._cur_task == 0 or self.a_multiplier == 1.:
            return
        if not isinstance(optimizer, torch.optim.SGD):
            raise ValueError('original SGD optimizer is required')
        with torch.no_grad():
            backbone = self._raw_network().backbone
            for module in list(backbone.w_Bs) + list(backbone.wrapped_param):
                for parameter in module.parameters():
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError('nonfinite current-B/scale gradient')
            for module in backbone.w_As:
                parameter = module.weight
                if parameter.grad is None:
                    continue
                groups = [group for group in optimizer.param_groups
                          if any(p is parameter for p in group['params'])]
                if len(groups) != 1:
                    raise RuntimeError('A must belong to one optimizer group')
                adjusted, record = amplify_projected_gradient(
                    parameter.grad, parameter, self.a_multiplier,
                    weight_decay=float(groups[0].get('weight_decay', 0.)))
                parameter.grad.copy_(adjusted)
                if record['nonzero']:
                    self._a_gain_stats['nonzero_scaled_branches'] += 1
                    self._a_gain_stats['projected_gradient_norm_before'] += record['before_norm']
                    self._a_gain_stats['projected_gradient_norm_after'] += record['after_norm']
                self._a_gain_stats['max_relative_direction_error'] = max(
                    self._a_gain_stats['max_relative_direction_error'], record['relative_error'])
        self._a_gain_stats['steps'] += 1

    def _before_task_save(self, raw_network, train_loader):
        super()._before_task_save(raw_network, train_loader)
        root = Path(self.args['filepath']) / 'a_gain'
        root.mkdir(exist_ok=True)
        record = dict(self._a_gain_stats, task=self._cur_task, multiplier=self.a_multiplier,
                      task0_unchanged=self._cur_task == 0,
                      transformation='k*(projected_gradient-decay*A)+ (k-1)*decay*A',
                      deployment_extra_parameters=0)
        before = record['projected_gradient_norm_before']
        record['observed_norm_multiplier'] = None if before == 0 else record['projected_gradient_norm_after'] / before
        (root / f'task_{self._cur_task:03d}.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
