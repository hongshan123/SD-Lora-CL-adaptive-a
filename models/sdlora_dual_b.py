"""SD-LoRA (per-task LoRA bank) with the same Dual-B prototype/FC head.

P4 parameter-matched baseline: ten rank-1 SD-LoRA adapters (LoRA bank
~368,640) plus the identical prototype-cosine + FC Dual-B schedule used by
the complete Live-A Aggregate-B + Dual-B method.  This isolates whether the
aggregate rank-10 online capacity beats ten rank-1 task adapters at the same
persistent state budget.
"""

import logging
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from models.sdlora import Learner as SDLoraLearner
from models.sa_sdlora import (
    all_ranks_equal,
    broadcast_dual_head_values,
    broadcast_prototypes,
    evaluate_dual_head_once,
)
from utils.canonical_hash import model_tensor_map, compare_named_tensors
from utils.inc_net import SharedAPrototypeNet
from utils.rng_utils import (
    deterministic_loader,
    rng_preserving,
    rng_state_hash,
)


num_workers = 8
PROTOTYPES_FILENAME = "sa_prototypes.pt"


class Learner(SDLoraLearner):
    """SD-LoRA bank + prototype classifier + fixed Dual-B schedule."""

    def __init__(self, args):
        super().__init__(args)
        self._dual_head = bool(args.get("sa_dual_head", False))
        self._dual_schedule = args.get("sa_dual_head_schedule", "A")
        if self._dual_schedule not in ("A", "B"):
            raise ValueError("sa_dual_head_schedule must be A or B")
        if not args.get("sa_use_prototype_classifier", False):
            raise ValueError(
                "sdlora_dual_b requires sa_use_prototype_classifier=True"
            )
        self._network = SharedAPrototypeNet(args, True)
        self._dual_num_tasks = 0

    def _wrap_ddp(self):
        # Per-task LoRA/scale parameters are genuinely unused on some ranks;
        # mirror the legacy SD-LoRA handling.
        if self._is_distributed():
            local_rank = self.args["local_rank"]
            self._network = DDP(
                self._network,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )

    def incremental_train(self, data_manager):
        self._dual_num_tasks = data_manager.nb_tasks
        if self._dual_head:
            raw_network = self._raw_network()
            raw_network.set_head_mode("fc")
            raw_network.dual_head = False
            raw_network.dual_lambda = 0.0
            raw_network.tau_fc = 1.0
            raw_network.tau_proto = 1.0
        super().incremental_train(data_manager)
        if self.args.get("sa_use_prototype_classifier", False):
            if self._is_main_process():
                prototypes = self._compute_prototypes(
                    data_manager, self._raw_network()
                )
            else:
                prototypes = {}
            prototypes = broadcast_prototypes(prototypes, src=0)
            raw_network = self._raw_network()
            raw_network.set_prototypes(prototypes)
            logging.info(
                "[SDLoRA-DualB] prototype classifier active for %d classes",
                len(prototypes),
            )
        if self._dual_head:
            self._prepare_dual_head(data_manager, self._raw_network())

    def _loader_workers(self):
        if self.args.get("sa_deterministic_training", False):
            return 0
        return num_workers

    def _extract_current_task_features(
        self, data_manager, return_targets=False
    ):
        """Extract current-task training features (test transform, no aug)."""
        cur_classes = np.arange(
            self._known_classes,
            self._known_classes + data_manager.get_task_size(self._cur_task),
        )
        dataset = data_manager.get_dataset(
            cur_classes, source="train", mode="test"
        )
        raw_network = self._raw_network()
        raw_network.eval()
        features, targets = [], []
        with rng_preserving(), torch.no_grad():
            loader = deterministic_loader(
                dataset,
                batch_size=64,
                shuffle=False,
                num_workers=self._loader_workers(),
                seed=0,
            )
            for _, inputs, batch_targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                feats = raw_network.backbone(inputs)
                features.append(feats.detach().cpu())
                targets.append(batch_targets)
        features = torch.cat(features, dim=0)
        if return_targets:
            return features, torch.cat(targets, dim=0)
        return features

    def _compute_prototypes(self, data_manager, raw_network):
        """Per-class L2-normalized mean prototypes from current-task data."""
        cur_classes = np.arange(self._known_classes, self._total_classes)
        dataset = data_manager.get_dataset(
            cur_classes, source="train", mode="test"
        )
        raw_network.eval()
        per_class = {int(c): [] for c in cur_classes}
        with rng_preserving(), torch.no_grad():
            loader = deterministic_loader(
                dataset,
                batch_size=64,
                shuffle=False,
                num_workers=self._loader_workers(),
                seed=0,
            )
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                feats = F.normalize(raw_network.backbone(inputs), p=2, dim=1)
                for f, target in zip(feats.cpu(), targets):
                    per_class[int(target.item())].append(f)
        prototypes = {
            c: F.normalize(torch.stack(per_class[c]).mean(dim=0), p=2, dim=0)
            for c in cur_classes.tolist()
        }
        path = os.path.join(self.args["filepath"], PROTOTYPES_FILENAME)
        if os.path.exists(path):
            old = torch.load(path, map_location="cpu", weights_only=True)
            for class_id, vector in old.items():
                prototypes.setdefault(int(class_id), vector)
        torch.save(prototypes, path)
        return prototypes

    def _dual_lambda(self, task_id, num_tasks):
        if num_tasks <= 1:
            return 1.0
        progress = task_id / (num_tasks - 1)
        width = 4.0 / 9.0
        start = 0.0 if self._dual_schedule == "A" else 1.0 / 9.0
        return min(1.0, max(0.0, (progress - start) / width))

    @staticmethod
    def _fit_dual_temperature(logits, targets, lo=0.05, hi=5.0, steps=60):
        best_tau, best_loss = 1.0, float("inf")
        for tau in torch.linspace(lo, hi, steps).tolist():
            loss = F.cross_entropy(logits / tau, targets).item()
            if loss < best_loss:
                best_loss, best_tau = loss, tau
        return best_tau

    def _prepare_dual_head(self, data_manager, raw_network):
        lambda_val, tau_fc, tau_proto = 0.0, 1.0, 1.0
        if self._is_main_process():
            raw_network.eval()
            before_map = model_tensor_map(raw_network)
            rng_before = rng_state_hash()
            with rng_preserving(), torch.no_grad():
                features, targets = self._extract_current_task_features(
                    data_manager, return_targets=True
                )
                features = features.to(self._device)
                targets = targets.to(self._device)
                fc_logits = raw_network.fc(features)["logits"]
                proto_logits = raw_network.prototype_head(features)["logits"]
            tau_fc = self._fit_dual_temperature(fc_logits, targets)
            tau_proto = self._fit_dual_temperature(proto_logits, targets)
            lambda_val = self._dual_lambda(
                self._cur_task, self._dual_num_tasks
            )
            mismatch = compare_named_tensors(
                before_map, model_tensor_map(raw_network)
            )
            if mismatch is not None:
                raise RuntimeError(
                    "Dual-head calibration mutated model tensors: first "
                    "mismatch {} (max_abs_diff={})".format(
                        mismatch["key"], mismatch["max_abs_diff"]
                    )
                )
            rng_after = rng_state_hash()
            if rng_before != rng_after:
                raise RuntimeError(
                    "Dual-head calibration perturbed RNG state: {} -> {}".format(
                        rng_before, rng_after
                    )
                )
            logging.info(
                "[RNGHash] task %d calibration before=%s after=%s PASS",
                self._cur_task,
                rng_before,
                rng_after,
            )
            logging.info(
                "[DualHead] calibration parameter invariance PASS "
                "(max_abs_diff=0.000e+00)"
            )
            state = {
                "schedule": self._dual_schedule,
                "task_id": self._cur_task,
                "num_tasks": self._dual_num_tasks,
                "lambda": lambda_val,
                "tau_fc": tau_fc,
                "tau_proto": tau_proto,
            }
            torch.save(
                state,
                os.path.join(self.args["filepath"], "sa_dual_head.pt"),
            )
            logging.info(
                "[DualHead] task %d schedule=%s lambda=%.3f tau_fc=%.3f "
                "tau_proto=%.3f",
                self._cur_task,
                self._dual_schedule,
                lambda_val,
                tau_fc,
                tau_proto,
            )
        lambda_val, tau_fc, tau_proto = broadcast_dual_head_values(
            lambda_val, tau_fc, tau_proto, src=0
        )
        raw_network.set_dual_head(lambda_val, tau_fc, tau_proto)
        if not all_ranks_equal(
            torch.tensor(
                [lambda_val, tau_fc, tau_proto], dtype=torch.float64
            )
        ):
            raise RuntimeError("dual-head values differ across ranks")

    def eval_task(self):
        if not self._is_main_process():
            return None, None
        rng_before = rng_state_hash()
        loader = deterministic_loader(
            self._eval_test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self._loader_workers(),
            seed=0,
        )
        # The SD-LoRA train wrapper instantiates fresh nn.Linear modules on
        # every forward (default init draws RNG), so the evaluation traversal
        # restores the RNG state; this keeps eval invisible to training.
        with rng_preserving():
            preds, y_true, fused_proto_diff = evaluate_dual_head_once(
                self._raw_network(),
                loader,
                device=self._device,
                topk=self.topk,
            )
        if float(self._raw_network().dual_lambda) >= 1.0:
            logging.info(
                "[DualHead] task %d fused_proto_max_diff=%.3e PASS",
                self._cur_task,
                fused_proto_diff,
            )
        rng_after = rng_state_hash()
        if rng_before != rng_after:
            raise RuntimeError(
                "dual eval perturbed RNG state: {} -> {}".format(
                    rng_before, rng_after
                )
            )
        logging.info(
            "[RNGHash] task %d eval before=%s after=%s PASS",
            self._cur_task,
            rng_before,
            rng_after,
        )
        accuracies = {}
        for mode in preds:
            accuracies[mode] = self._evaluate(
                np.concatenate(preds[mode]), np.concatenate(y_true)
            )
            logging.info(
                "[DualHead] task %d mode=%s top1=%.2f top5=%.2f",
                self._cur_task,
                mode,
                accuracies[mode]["top1"],
                accuracies[mode]["top5"],
            )
        return accuracies["fused"], None
