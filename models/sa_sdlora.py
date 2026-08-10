import logging
import copy
import json
import math
import os

import numpy as np
import timm
import torch
import torch.distributed as dist
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from backbone.lrpt import (
    apply_transport,
    apply_transport_adaptive,
    bias_relative_error,
    fit_affine_low_rank_transport,
    fit_affine_map,
    fit_affine_rank_residuals,
    fit_layerwise_weights,
    fit_low_rank_transport,
    fit_rank_residuals,
    low_rank_factors,
    project_map_to_basis,
    transport_prediction_error,
)
from backbone.sa_lora import (
    SharedALoRA_ViT_timm,
    hbd_historical_branch_distance,
    live_a_historical_outputs,
    register_live_a_historical_capture_hooks,
)
from utils.canonical_hash import (
    compare_named_tensors,
    hash_named_tensors,
    model_tensor_map,
)
from utils.inc_net import SimpleCosineIncrementalNet, SharedAPrototypeNet
from utils.rng_utils import (
    deterministic_loader,
    rng_preserving,
    rng_state_hash,
)
from models.sdlora import Learner as SDLoraLearner


num_workers = 8
PROTOTYPES_FILENAME = "sa_prototypes.pt"
P0_HASHES_FILENAME = "p0_hashes.json"


def collective_device():
    """Return the tensor device required by the current process group backend.

    NCCL collectives require CUDA tensors; Gloo accepts CPU tensors.
    """
    if dist.is_initialized():
        backend = dist.get_backend()
        if backend == dist.Backend.NCCL:
            return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def dual_head_logits(network, features):
    """Return calibrated fc/prototype/fused logits for one feature batch."""
    fc_logits = network.fc(features)["logits"] / network.tau_fc
    proto_logits = (
        network.prototype_head(features)["logits"] / network.tau_proto
    )
    fused_logits = (
        (1.0 - network.dual_lambda) * fc_logits
        + network.dual_lambda * proto_logits
    )
    return fc_logits, proto_logits, fused_logits


def serialize_prototypes(prototypes):
    """Flatten a prototype dict into a broadcastable tensor + metadata."""
    if not prototypes:
        return [0], torch.zeros(1, 1, 1), torch.zeros(1, dtype=torch.long)
    keys = sorted(int(key) for key in prototypes)
    first = prototypes[keys[0]]
    if isinstance(first, (list, tuple)):
        max_k = max(len(prototypes[key]) for key in keys)
        dim = first[0].shape[0]
        stacked = torch.full(
            (len(keys), max_k, dim),
            float("nan"),
            dtype=first[0].dtype,
        )
        counts = torch.zeros(len(keys), dtype=torch.long)
        for index, key in enumerate(keys):
            vectors = prototypes[key]
            counts[index] = len(vectors)
            for vector_index, vector in enumerate(vectors):
                stacked[index, vector_index] = vector
    else:
        stacked = torch.stack([prototypes[key] for key in keys]).unsqueeze(1)
        counts = torch.ones(len(keys), dtype=torch.long)
    return keys, stacked, counts


def deserialize_prototypes(keys, stacked, counts):
    """Rebuild a prototype dict from broadcasted metadata."""
    prototypes = {}
    for index, key in enumerate(keys):
        key = int(key)
        num = int(counts[index].item()) if counts.numel() else 1
        if num > 1:
            prototypes[key] = [
                stacked[index, vector_index].detach().clone()
                for vector_index in range(num)
            ]
        else:
            prototypes[key] = stacked[index, 0].detach().clone()
    return prototypes


def broadcast_prototypes(prototypes, src=0):
    """Broadcast prototype tensors from ``src`` to every DDP rank."""
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return prototypes
    keys, stacked, counts = serialize_prototypes(prototypes)
    rank = dist.get_rank()
    device = collective_device()
    size_tensor = torch.zeros(4, dtype=torch.long, device=device)
    if rank == src:
        size_tensor = torch.tensor(
            [len(keys), stacked.shape[0], stacked.shape[1], stacked.shape[2]],
            dtype=torch.long,
            device=device,
        )
    dist.broadcast(size_tensor, src=src)
    flat = torch.zeros(
        int(size_tensor[1] * size_tensor[2] * size_tensor[3]),
        dtype=torch.float32,
        device=device,
    )
    if rank == src:
        flat = stacked.to(device=device, dtype=torch.float32).reshape(-1)
    dist.broadcast(flat, src=src)
    stacked = flat.cpu().reshape(
        int(size_tensor[1]), int(size_tensor[2]), int(size_tensor[3])
    )
    keys_tensor = torch.zeros(int(size_tensor[0]), dtype=torch.long, device=device)
    if rank == src:
        keys_tensor = torch.tensor(keys, dtype=torch.long, device=device)
    dist.broadcast(keys_tensor, src=src)
    broadcast_counts = torch.zeros(
        int(size_tensor[0]), dtype=torch.long, device=device
    )
    if rank == src:
        broadcast_counts = counts.to(device=device)
    dist.broadcast(broadcast_counts, src=src)
    return deserialize_prototypes(
        keys_tensor.tolist(), stacked, broadcast_counts.cpu()
    )


def broadcast_dual_head_values(lambda_val, tau_fc, tau_proto, src=0):
    """Broadcast calibration scalars from ``src`` to every DDP rank."""
    values = torch.tensor(
        [lambda_val, tau_fc, tau_proto],
        dtype=torch.float64,
        device=collective_device(),
    )
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast(values, src=src)
    return float(values[0]), float(values[1]), float(values[2])


def all_ranks_equal(value):
    """True when ``value`` is bit-identical on every DDP rank."""
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return True
    device = collective_device()
    flat = value.detach().to(device=device, dtype=torch.float64).reshape(-1)
    gathered = [
        torch.zeros_like(flat) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered, flat)
    return all(torch.equal(item, flat) for item in gathered)


def evaluate_dual_head_once(network, loader, device=None, topk=5):
    """One DataLoader traversal computing fc/proto/fused logits and top-k preds.

    Returns ``(preds, y_true, max_fused_proto_diff)``.  When ``lambda==1`` the
    fused logits must equal prototype logits up to 1e-6; a violation raises
    ``RuntimeError`` with the concrete diff.
    """
    network.eval()
    preds = {"fc": [], "proto": [], "fused": []}
    y_true = []
    max_fused_proto_diff = 0.0
    with torch.no_grad():
        for _, inputs, targets in loader:
            if device is not None:
                inputs = inputs.to(device, non_blocking=True)
            features = network.backbone(inputs)
            fc_logits, proto_logits, fused_logits = dual_head_logits(
                network, features
            )
            if float(network.dual_lambda) >= 1.0:
                fused_proto_diff = (
                    (fused_logits - proto_logits).abs().max().item()
                )
                max_fused_proto_diff = max(
                    max_fused_proto_diff, fused_proto_diff
                )
                if fused_proto_diff > 1e-6:
                    raise RuntimeError(
                        "lambda=1 fused/prototype logits differ by "
                        "{:.3e} (max allowed 1e-6)".format(
                            fused_proto_diff
                        )
                    )
            for mode, logits in (
                ("fc", fc_logits),
                ("proto", proto_logits),
                ("fused", fused_logits),
            ):
                topk_indices = torch.topk(
                    logits,
                    k=topk,
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices.cpu().numpy()
                preds[mode].append(topk_indices)
            y_true.append(targets.cpu().numpy())
    return preds, y_true, max_fused_proto_diff


class SharedACosineNet(SimpleCosineIncrementalNet):
    """SimpleCosineIncrementalNet with SD-LoRA-style fc artifact saving."""

    def save_fc(self, filename, task_id):
        torch.save(
            self.fc.weight.detach(), filename + "CLs_weight" + str(task_id) + ".pt"
        )
        torch.save(
            torch.zeros(self.fc.out_features),
            filename + "CLs_bias" + str(task_id) + ".pt",
        )


class Learner(SDLoraLearner):
    """Shared-A SD-LoRA: task-invariant A, per-task B, exact final merging."""

    def __init__(self, args):
        super().__init__(args)
        use_cosine = args.get("sa_use_cosine_head", False)
        use_prototypes = args.get("sa_use_prototype_classifier", False)
        if use_cosine and use_prototypes:
            raise ValueError(
                "sa_use_cosine_head and sa_use_prototype_classifier "
                "are mutually exclusive"
            )
        if use_cosine:
            self._network = SharedACosineNet(args, True)
        elif use_prototypes:
            self._network = SharedAPrototypeNet(args, True)
        self._dual_head = bool(args.get("sa_dual_head", False))
        self._dual_schedule = args.get("sa_dual_head_schedule", "A")
        if self._dual_head and not use_prototypes:
            raise ValueError(
                "sa_dual_head requires sa_use_prototype_classifier=True"
            )
        if self._dual_schedule not in ("A", "B"):
            raise ValueError("sa_dual_head_schedule must be A or B")
        if self._dual_head and int(args.get("sa_k_prototypes", 1)) > 1:
            raise ValueError(
                "sa_dual_head and sa_k_prototypes>1 are mutually exclusive"
            )
        self._dual_num_tasks = 0
        self._lrpt_enabled = bool(args.get("lrpt_enabled", False))
        if self._lrpt_enabled and not use_prototypes:
            raise ValueError(
                "lrpt_enabled requires sa_use_prototype_classifier=True"
            )
        self._lrpt_rank = int(args.get("lrpt_rank", args.get("lora_rank", 10)))
        self._lrpt_reg = float(args.get("lrpt_reg", 1e-2))
        self._lrpt_bias = bool(args.get("lrpt_bias", False))
        self._lrpt_damping = float(args.get("lrpt_damping", 1.0))
        self._lrpt_dual = bool(args.get("lrpt_dual", False))
        self._lrpt_class_weight = float(args.get("lrpt_class_weight", 0.3))
        self._lrpt_basis = args.get("lrpt_basis", "generic")
        if self._lrpt_basis not in ("generic", "delta_a", "delta_a_b", "layerwise"):
            raise ValueError(
                "lrpt_basis must be generic/delta_a/delta_a_b/layerwise"
            )
        self._lrpt_basis_rank = int(
            args.get("lrpt_basis_rank", self._lrpt_rank)
        )
        self._lrpt_basis_max_samples = int(
            args.get("lrpt_basis_max_samples", 4096)
        )
        self._lrpt_basis_cap = int(args.get("lrpt_basis_cap", 128))
        self._lrpt_adaptive = bool(args.get("lrpt_adaptive", False))
        self._lrpt_fit_target = args.get("lrpt_fit_target", "sample")
        if self._lrpt_fit_target not in ("sample", "classmean"):
            raise ValueError(
                "lrpt_fit_target must be 'sample' or 'classmean'"
            )
        self._lrpt_diagnostics = bool(args.get("lrpt_diagnostics", True))
        self._lrpt_pre_features = None
        self._lrpt_pre_targets = None
        self._lrpt_a_old = None
        self._lrpt_res_pre = None
        self._sa_operator_stability_lambda = float(
            args.get("sa_operator_stability_lambda", 0.0)
        )
        self._sa_prototype_consistency_weight = float(
            args.get("sa_prototype_consistency_weight", 0.0)
        )
        self._hbd_enabled = bool(args.get("sa_hbd_enabled", False))
        self._hbd_lambda = float(args.get("sa_hbd_lambda", 0.1))
        if self._hbd_enabled:
            if self._hbd_lambda <= 0:
                raise ValueError(
                    "sa_hbd_lambda must be > 0 when sa_hbd_enabled=true"
                )
            if args.get("sa_cumulative_merge", "gauge") != "live_a_aggregate_b":
                raise ValueError(
                    "sa_hbd_enabled requires "
                    "sa_cumulative_merge=live_a_aggregate_b"
                )
        self._hbd_teacher = None
        self._hbd_teacher_captures = []
        self._hbd_teacher_handles = []
        self._hbd_first_batch = True
        self._proto_ema = {}
        self._proto_ema_task = None
        if self._sa_operator_stability_lambda < 0:
            raise ValueError("sa_operator_stability_lambda must be non-negative")
        if (
            self._sa_operator_stability_lambda > 0
            and not args.get("sa_train_a_all_tasks", False)
        ):
            raise ValueError(
                "sa_operator_stability_lambda requires sa_train_a_all_tasks=True"
            )
        if (
            self._sa_operator_stability_lambda > 0
            and args.get("sa_cumulative_state", False)
        ):
            raise ValueError(
                "operator-stability loss is incompatible with cumulative "
                "Shared-A state (route closed; use gauge residual diagnostics)"
            )

    def update_network(self, index=True, task_index=None):
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )
        cur_task_index = self._cur_task if task_index is None else task_index
        model = SharedALoRA_ViT_timm(
            vit_model=model.eval(),
            r=self.args.get("lora_rank", 10),
            num_classes=10,
            index=index,
            increment=self.args["increment"],
            filepath=self.args["filepath"],
            cur_task_index=cur_task_index,
            shared_a_orthogonal=self.args.get("sa_shared_a_orthogonal", True),
            train_a_all_tasks=self.args.get("sa_train_a_all_tasks", False),
            delete_per_task_files=self.args.get("sa_delete_per_task_files", False),
            cumulative_state=self.args.get("sa_cumulative_state", False),
            cumulative_gauge=self.args.get("sa_cumulative_gauge", True),
            cumulative_merge=self.args.get("sa_cumulative_merge", "gauge"),
            cumulative_rank=self.args.get("sa_cumulative_rank", None),
            freeze_old_scales=self.args.get("sa_freeze_old_scales", False),
            live_a_history_groups=self.args.get("sa_live_a_history_groups", 1),
            resume=self.args.get("sa_resume", False),
        )
        model.out_dim = 768
        return model

    def incremental_train(self, data_manager):
        self._dual_num_tasks = data_manager.nb_tasks
        if self._hbd_enabled and self._cur_task >= 0:
            backbone = self._raw_network().backbone
            self._hbd_teacher = backbone.build_hbd_teacher()
            self._hbd_teacher_captures = []
            self._hbd_teacher_handles = (
                register_live_a_historical_capture_hooks(
                    self._hbd_teacher, self._hbd_teacher_captures
                )
            )
            self._hbd_first_batch = True
        if self._dual_head:
            raw_network = self._raw_network()
            raw_network.set_head_mode("fc")
            # Calibration/eval scalars are eval-only; do not let them persist
            # into the next task's training-state hash.
            raw_network.dual_head = False
            raw_network.dual_lambda = 0.0
            raw_network.tau_fc = 1.0
            raw_network.tau_proto = 1.0
        # Capture the current task's features in the *previous* model state
        # before Shared-A is updated. Only current-task data is touched.
        if (
            self._lrpt_enabled
            and self._is_main_process()
            and self._cur_task >= 0
        ):
            self._lrpt_pre_features, self._lrpt_pre_targets = (
                self._extract_current_task_features(
                    data_manager,
                    task_index=self._cur_task + 1,
                    return_targets=True,
                )
            )
            raw_network = self._raw_network()
            self._lrpt_a_old = [
                w.weight.detach().cpu().clone()
                for w in raw_network.backbone.w_As
            ]
            logging.info(
                "[SharedA-SDLoRA] LRPT pre-update features captured for task %d (%d samples)",
                self._cur_task + 1,
                self._lrpt_pre_features.shape[0],
            )
        super().incremental_train(data_manager)
        if self._hbd_teacher is not None:
            for handle in self._hbd_teacher_handles:
                handle.remove()
            self._hbd_teacher_handles = []
            self._hbd_teacher_captures = []
            self._hbd_teacher = None
        if self._is_main_process():
            self._log_post_train_hash(self._cur_task)
        if self._is_main_process():
            if self._cur_task > 0:
                with torch.no_grad():
                    backbone = self._raw_network().backbone
                    if backbone.cumulative_state:
                        if (
                            backbone.cumulative_merge
                            == "live_a_aggregate_b"
                        ):
                            stats = getattr(
                                backbone, "_last_live_a_save_stats", None
                            )
                            if stats is not None:
                                logging.info(
                                    "[LiveA-SDLoRA] save task %d: "
                                    "mean_G=%.4e mean_A=%.4e mean_B=%.4e "
                                    "scale=%.4f",
                                    self._cur_task,
                                    stats["mean_G_norm"],
                                    stats["mean_A_norm"],
                                    stats["mean_B_norm"],
                                    stats["scale"],
                                )
                        elif backbone.cumulative_merge == "union_svd":
                            trunc_error = getattr(
                                backbone,
                                "_last_union_svd_truncation_error",
                                None,
                            )
                            logging.info(
                                "[SharedA-SDLoRA] union-svd task %d: "
                                "max_relative_truncation_error=%.6e",
                                self._cur_task,
                                trunc_error
                                if trunc_error is not None
                                else float("nan"),
                            )
                        else:
                            gauge_diag = getattr(
                                backbone,
                                "_last_cumulative_gauge_diagnostics",
                                None,
                            )
                            if gauge_diag is None:
                                logging.warning(
                                    "[SharedA-SDLoRA] pre-save cumulative "
                                    "gauge diagnostics missing; logging zeros"
                                )
                                gauge_diag = {
                                    "residual": 0.0,
                                    "rotation_fro": 0.0,
                                    "preservation": 0.0,
                                    "branches": 0,
                                }
                            logging.info(
                                "[SharedA-SDLoRA] cumulative gauge task %d: "
                                "relative_projection_residual=%.6e "
                                "basis_rotation_fro=%.6e "
                                "operator_preservation=%.6e",
                                self._cur_task,
                                gauge_diag["residual"],
                                gauge_diag["rotation_fro"],
                                gauge_diag["preservation"],
                            )
                    else:
                        operator_drift = float(
                            backbone.old_operator_stability_loss().item()
                        )
                        logging.info(
                            "[SharedA-SDLoRA] operator-stability task %d: "
                            "relative_effective_drift=%.6f lambda=%.4g",
                            self._cur_task,
                            operator_drift,
                            self._sa_operator_stability_lambda,
                        )
            if self._cur_task == data_manager.nb_tasks - 1:
                self._rebuild_eval_backbone()
            raw_network = self._raw_network()
            raw_network.backbone.save_merged_lora(self.args["filepath"])
            logging.info(
                "[SharedA-SDLoRA] saved merged LoRA after task %d",
                self._cur_task,
            )
            if self._cur_task == data_manager.nb_tasks - 1:
                if self.args.get("sa_final_head_align", False):
                    raw_network.weight_align(int(self.args["increment"]))
                    logging.info("[SharedA-SDLoRA] final classifier weight aligned")
                tune_epochs = int(self.args.get("sa_final_head_tune_epochs", 0))
                if tune_epochs > 0:
                    self._final_head_tune(
                        data_manager,
                        raw_network,
                        epochs=tune_epochs,
                        lr=float(self.args.get("sa_final_head_lr", 1e-3)),
                    )
        if self.args.get("sa_use_prototype_classifier", False):
            if self._is_main_process():
                if (
                    self._lrpt_enabled
                    and self._lrpt_pre_features is not None
                ):
                    self._apply_lrpt_to_old_prototypes(data_manager)
                prototypes = self._compute_prototypes(
                    data_manager, raw_network
                )
            else:
                prototypes = {}
            prototypes = broadcast_prototypes(prototypes, src=0)
            raw_network = self._raw_network()
            raw_network.set_prototypes(prototypes)
            prototype_head = raw_network.prototype_head
            if prototype_head is not None and hasattr(
                prototype_head, "weight"
            ):
                if not all_ranks_equal(prototype_head.weight):
                    raise RuntimeError(
                        "prototype head differs across ranks after broadcast"
                    )
            if self._is_main_process():
                logging.info(
                    "[SharedA-SDLoRA] prototype classifier active for %d classes",
                    len(prototypes),
                )
                logging.info("[PrototypeSync] rank consistency PASS")
                raw_network.backbone.cleanup_per_task_files(
                    self.args["filepath"]
                )
        if self._dual_head:
            self._prepare_dual_head(data_manager, self._raw_network())

    def _p0_hash_store(self):
        """JSON file holding per-task tensor/RNG hashes for cross-run audits."""
        return os.path.join(self.args["filepath"], P0_HASHES_FILENAME)

    def _store_p0_hash(self, task_id, kind, model_hash, rng_hash):
        path = self._p0_hash_store()
        data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        data.setdefault(str(task_id), {})[kind + "_model_hash"] = model_hash
        data[str(task_id)][kind + "_rng_hash"] = rng_hash
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _log_post_train_hash(self, task_id):
        """Canonical tensor/RNG hash immediately after training, before eval."""
        raw_network = self._raw_network()
        tensors = model_tensor_map(raw_network, include_eval_scalars=False)
        model_hash = hash_named_tensors(tensors)
        rng_hash = rng_state_hash()
        self._store_p0_hash(task_id, "post_train", model_hash, rng_hash)
        logging.info(
            "[PostTrainHash] task %d hash=%s rng=%s",
            task_id,
            model_hash,
            rng_hash,
        )

    def _log_post_eval_hash(self, task_id):
        """Canonical tensor/RNG hash after calibration and evaluation."""
        raw_network = self._raw_network()
        tensors = model_tensor_map(raw_network, include_eval_scalars=False)
        model_hash = hash_named_tensors(tensors)
        rng_hash = rng_state_hash()
        self._store_p0_hash(task_id, "post_eval", model_hash, rng_hash)
        logging.info(
            "[PostEvalHash] task %d hash=%s rng=%s",
            task_id,
            model_hash,
            rng_hash,
        )

    def _assert_eval_tensor_invariance(self, before_map, task_id):
        after_map = model_tensor_map(self._raw_network())
        mismatch = compare_named_tensors(before_map, after_map)
        if mismatch is not None:
            raise RuntimeError(
                "eval mutated model tensors: first mismatch {} "
                "(max_abs_diff={})".format(
                    mismatch["key"], mismatch["max_abs_diff"]
                )
            )
        logging.info(
            "[EvalTensorHash] task %d before=%s after=%s PASS",
            task_id,
            hash_named_tensors(before_map),
            hash_named_tensors(after_map),
        )

    def eval_task(self):
        if not self._is_main_process():
            return None, None
        if not self._dual_head:
            before_map = model_tensor_map(self._raw_network())
            rng_before = rng_state_hash()
            result = super().eval_task()
            rng_after = rng_state_hash()
            if rng_before != rng_after:
                raise RuntimeError(
                    "control eval perturbed RNG state: {} -> {}".format(
                        rng_before, rng_after
                    )
                )
            logging.info(
                "[RNGHash] task %d eval before=%s after=%s PASS",
                self._cur_task,
                rng_before,
                rng_after,
            )
            self._assert_eval_tensor_invariance(before_map, self._cur_task)
            self._log_post_eval_hash(self._cur_task)
            return result

        before_map = model_tensor_map(self._raw_network())
        rng_before = rng_state_hash()
        loader = deterministic_loader(
            self._eval_test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self._loader_workers(),
            seed=0,
        )
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
        self._assert_eval_tensor_invariance(before_map, self._cur_task)
        if fused_proto_diff > 0.0:
            logging.info(
                "[DualHead] task %d fused_proto_max_diff=%.3e PASS",
                self._cur_task,
                fused_proto_diff,
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
        self._log_post_eval_hash(self._cur_task)
        return accuracies["fused"], None

    def _dual_lambda(self, task_id, num_tasks):
        if num_tasks <= 1:
            return 1.0
        progress = task_id / (num_tasks - 1)
        width = 4.0 / 9.0
        start = 0.0 if self._dual_schedule == "A" else 1.0 / 9.0
        return min(1.0, max(0.0, (progress - start) / width))

    def _loader_workers(self):
        if self.args.get("sa_deterministic_training", False):
            return 0
        return num_workers

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
                if (
                    getattr(self, "_last_proto_task", None) == self._cur_task
                    and getattr(self, "_last_proto_features", None) is not None
                ):
                    features = self._last_proto_features
                    targets = self._last_proto_targets
                else:
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
        lambda_val, tau_fc, tau_proto = broadcast_dual_head_values(
            lambda_val, tau_fc, tau_proto, src=0
        )
        raw_network.set_dual_head(lambda_val, tau_fc, tau_proto)
        if not all_ranks_equal(
            torch.tensor(
                [lambda_val, tau_fc, tau_proto], dtype=torch.float64
            )
        ):
            raise RuntimeError(
                "dual-head calibration scalars differ across ranks"
            )
        print(
            "[DualHead] rank {} task {} lambda={:.6f} tau_fc={:.6f} "
            "tau_proto={:.6f}".format(
                self.args.get("rank", 0),
                self._cur_task,
                lambda_val,
                tau_fc,
                tau_proto,
            ),
            flush=True,
        )
        if self._is_main_process():
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

    def _additional_training_losses(
        self, inputs=None, targets=None, features=None
    ):
        losses = {}
        if self._hbd_enabled and self._cur_task > 0:
            hbd_loss = self._hbd_training_loss(
                inputs, targets, features
            )
            if hbd_loss is not None:
                losses["hbd"] = hbd_loss
        if self._sa_operator_stability_lambda > 0 and self._cur_task > 0:
            operator_loss = (
                self._raw_network().backbone.old_operator_stability_loss()
            )
            losses["operator_stability"] = (
                self._sa_operator_stability_lambda * operator_loss
            )
        w = self._sa_prototype_consistency_weight
        if w <= 0 or features is None or targets is None:
            return losses
        if self._proto_ema_task != self._cur_task:
            self._proto_ema = {}
            self._proto_ema_task = self._cur_task
        feats = F.normalize(features.float(), p=2, dim=1)
        momentum = 0.1
        with torch.no_grad():
            for f, t in zip(feats, targets):
                c = int(t.item())
                if c not in self._proto_ema:
                    self._proto_ema[c] = f.clone()
                else:
                    self._proto_ema[c] = F.normalize(
                        (1.0 - momentum) * self._proto_ema[c]
                        + momentum * f,
                        p=2,
                        dim=0,
                    )
        proto_matrix = torch.stack(
            [self._proto_ema[int(t.item())] for t in targets]
        ).to(feats.device)
        cosine = (feats * proto_matrix).sum(dim=1)
        losses["prototype_consistency"] = w * (1.0 - cosine).mean()
        return losses

    def _hbd_training_loss(
        self, inputs=None, targets=None, features=None
    ):
        """Historical-Branch Activation Distillation loss (task t > 0).

        Runs one extra autograd forward over the live backbone to collect the
        per-block historical-branch responses (``G_prev * normalize(A_live) * x``)
        and compares them with the frozen task-start teacher responses
        (``G_prev * normalize(A_prev) * x_teacher``).  Only the historical
        branch is constrained; fresh ``B_t`` receives no HBD gradient.
        """
        if not self._hbd_enabled or self._cur_task == 0:
            return None
        if self._hbd_teacher is None:
            raise RuntimeError(
                "HBD teacher missing for task {}; snapshot must be created "
                "before training".format(self._cur_task)
            )
        raw_network = self._raw_network()
        backbone = raw_network.backbone
        student_captures = []
        handles = register_live_a_historical_capture_hooks(
            backbone, student_captures
        )
        try:
            with torch.enable_grad():
                backbone(inputs)
        finally:
            for handle in handles:
                handle.remove()
        with torch.no_grad():
            teacher_outputs = live_a_historical_outputs(
                self._hbd_teacher, inputs, self._hbd_teacher_captures
            )
        distance = hbd_historical_branch_distance(
            student_captures, teacher_outputs
        )
        hbd_loss = self._hbd_lambda * distance
        if self._hbd_first_batch:
            self._hbd_first_batch = False
            self._log_hbd_first_batch(
                raw_network, inputs, targets, features
            )
        return hbd_loss

    def _log_hbd_first_batch(
        self, raw_network, inputs, targets, features
    ):
        """Record CE/HBD values and the CE-vs-HBD gradient norm ratio on A.

        The ratio is computed on a deep-copied network so the diagnostic
        backwards never trigger DDP reduction hooks on the live parameters.
        RNG is restored afterwards so the diagnostic does not perturb the
        training trajectory.
        """
        if features is None:
            return
        with rng_preserving():
            clone = copy.deepcopy(raw_network)
            clone.train()
            clone_feats = clone.backbone(inputs)
            clone_logits = clone.fc(clone_feats)["logits"]
            ce_loss = F.cross_entropy(
                clone_logits[:, self._known_classes :],
                targets - self._known_classes,
            )
            clone_captures = []
            clone_handles = register_live_a_historical_capture_hooks(
                clone.backbone, clone_captures
            )
            try:
                with torch.enable_grad():
                    clone.backbone(inputs)
            finally:
                for handle in clone_handles:
                    handle.remove()
            with torch.no_grad():
                teacher_outputs = live_a_historical_outputs(
                    self._hbd_teacher, inputs, self._hbd_teacher_captures
                )
            clone_distance = hbd_historical_branch_distance(
                clone_captures, teacher_outputs
            )
            clone_hbd_loss = self._hbd_lambda * clone_distance
            a_params = [w.weight for w in clone.backbone.w_As]
            ce_norm, hbd_norm = 0.0, 0.0
            for parameter in a_params:
                parameter.grad = None
            ce_loss.backward(retain_graph=True)
            ce_norm = sum(
                p.grad.detach().square().sum().item() for p in a_params
            ) ** 0.5
            for parameter in a_params:
                parameter.grad = None
            clone_hbd_loss.backward(retain_graph=True)
            hbd_norm = sum(
                p.grad.detach().square().sum().item() for p in a_params
            ) ** 0.5
            for parameter in a_params:
                parameter.grad = None
        if self._is_main_process():
            logging.info(
                "[HBD] task %d first batch: CE=%.4f HBD=%.4f "
                "dL/dA_ce=%.4e dL/dA_hbd=%.4e ratio_hbd_ce=%.4f",
                self._cur_task,
                ce_loss.detach().item(),
                clone_hbd_loss.detach().item(),
                ce_norm,
                hbd_norm,
                hbd_norm / max(ce_norm, 1e-12),
            )

    def _extract_current_task_features(
        self, data_manager, task_index=None, normalize=False, return_targets=False
    ):
        """Extract features of the current task's training data (test transform,
        no augmentation) with the current model state. By default returns raw
        features; pass normalize=True for the L2-normalized classifier space."""
        if task_index is None:
            task_index = self._cur_task
        cur_classes = np.arange(
            self._known_classes,
            self._known_classes + data_manager.get_task_size(task_index),
        )
        dataset = data_manager.get_dataset(
            cur_classes, source="train", mode="test"
        )
        raw_network = self._raw_network()
        raw_network.eval()
        features = []
        targets = []
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
                if normalize:
                    feats = F.normalize(feats, p=2, dim=1)
                features.append(feats.detach().cpu())
                targets.append(batch_targets)
        features = torch.cat(features, dim=0)
        if return_targets:
            return features, torch.cat(targets, dim=0)
        return features

    def _collect_anchor_images(self, data_manager, task_index=None, per_class=2):
        """Collect a few real current-task images per class as anchors for the
        LoRA parameter JVP (J_A(x) ΔA + J_B(x) ΔB)."""
        if task_index is None:
            task_index = self._cur_task
        cur_classes = np.arange(
            self._known_classes,
            self._known_classes + data_manager.get_task_size(task_index),
        )
        dataset = data_manager.get_dataset(
            cur_classes, source="train", mode="test"
        )
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        selected = {int(c): [] for c in cur_classes.tolist()}
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                for img, target in zip(inputs, targets):
                    c = int(target.item())
                    if len(selected[c]) < per_class:
                        selected[c].append(img)
                if all(len(v) >= per_class for v in selected.values()):
                    break
        anchors = torch.stack(
            [img for values in selected.values() for img in values]
        )
        return anchors

    def _lora_jvp_basis(self, data_manager, mode="delta_a_b"):
        """Compute the final-feature drift responses induced by the actual LoRA
        parameter changes (ΔA and/or new B_t) via JVP on current-task anchors,
        then take the top-r right singular vectors as the transport basis."""
        from torch.func import functional_call
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

        anchors = self._collect_anchor_images(data_manager, per_class=2)
        module = self._raw_network().backbone
        module.eval()
        a_weights = [w.weight for w in module.w_As]
        b_weights = [w.weight for w in module.w_Bs]

        # Match LoRA parameter names by object identity.
        name_to_weight = {}
        for name, param in module.named_parameters():
            for weight in a_weights + b_weights:
                if param is weight and name not in name_to_weight:
                    name_to_weight[name] = weight
                    break
        order = []
        for weight in a_weights + b_weights:
            for name, mapped in name_to_weight.items():
                if mapped is weight and name not in order:
                    order.append(name)
                    break
        lora_names = set(order)
        base_params = {
            name: param.detach()
            for name, param in module.named_parameters()
            if name not in lora_names
        }

        a_new = [w.detach().clone().to(self._device) for w in a_weights]
        b_cur = [w.detach().clone().to(self._device) for w in b_weights]
        a_old = [t.to(self._device) for t in self._lrpt_a_old]
        delta_a = [a_new[i] - a_old[i] for i in range(len(a_new))]
        lora_cur = [t.clone() for t in a_new + b_cur]

        def compute_jvp(v_vec, anchor_batch):
            with sdpa_kernel(SDPBackend.MATH):
                with dual_level():
                    dual_params = dict(base_params)
                    for name, primal, tangent in zip(
                        order, lora_cur, v_vec
                    ):
                        dual_params[name] = make_dual(primal, tangent)
                    out = functional_call(
                        module, dual_params, anchor_batch
                    )
                    return unpack_dual(out).tangent.detach()

        if mode == "layerwise":
            layer_bases = []
            num_a = len(a_new)
            for layer_id in range(12):
                v = [torch.zeros_like(t) for t in lora_cur]
                for offset in (2 * layer_id, 2 * layer_id + 1):
                    v[offset] = delta_a[offset]
                    v[num_a + offset] = b_cur[offset]
                jvp_out = torch.cat(
                    [compute_jvp(v, chunk) for chunk in anchors.split(8)],
                    dim=0,
                )
                _, _, vh = torch.linalg.svd(
                    jvp_out.float(), full_matrices=False
                )
                layer_basis = (
                    vh[: self._lrpt_basis_rank].t().contiguous().cpu()
                )
                layer_bases.append(layer_basis)
            return layer_bases

        if mode == "delta_a":
            v = delta_a + [torch.zeros_like(t) for t in b_cur]
        elif mode == "delta_a_b":
            v = delta_a + b_cur
        else:
            raise ValueError("unknown LoRA basis mode {}".format(mode))
        jvp_out = torch.cat(
            [compute_jvp(v, chunk) for chunk in anchors.split(8)],
            dim=0,
        )
        _, _, vh = torch.linalg.svd(jvp_out.float(), full_matrices=False)
        basis = vh[: self._lrpt_basis_rank].t().contiguous().cpu()
        return basis

    def _fit_lora_aware_transport(self, data_manager, fit_old, fit_new):
        """Fit a transport constrained to the final-feature drift subspace
        induced by the real LoRA parameter changes (JVP-based)."""
        try:
            if self._lrpt_a_old is None:
                raise RuntimeError("missing pre-update shared-A capture")
            w_emp, bias = fit_affine_map(
                fit_old,
                fit_new,
                reg=self._lrpt_reg,
            )

            if self._lrpt_basis == "layerwise":
                layer_bases = self._lora_jvp_basis(
                    data_manager, mode="layerwise"
                )
                layer_maps = [
                    project_map_to_basis(w_emp, layer_basis)
                    for layer_basis in layer_bases
                ]
                alpha = fit_layerwise_weights(
                    fit_old,
                    fit_new,
                    layer_maps,
                    reg=self._lrpt_reg,
                )
                w = torch.zeros_like(w_emp)
                for a_l, w_l in zip(alpha.tolist(), layer_maps):
                    w = w + a_l * w_l
                logging.info(
                    "[SharedA-SDLoRA] LRPT layerwise alpha=%s",
                    [round(float(a), 4) for a in alpha.tolist()],
                )
            else:
                basis = self._lora_jvp_basis(
                    data_manager, mode=self._lrpt_basis
                )
                w = project_map_to_basis(w_emp, basis)
                logging.info(
                    "[SharedA-SDLoRA] LRPT basis=%s dims=%d",
                    self._lrpt_basis,
                    basis.shape[1],
                )

            u, v = low_rank_factors(w, rank=self._lrpt_rank)
            self._lrpt_a_old = None
            return u, v, bias
        except Exception:
            logging.exception("[SharedA-SDLoRA] LRPT LoRA-aware transport failed")
            raise

    def _apply_lrpt_to_old_prototypes(self, data_manager):
        """Fit the low-rank transport from the paired current-task features and
        recursively move all previously stored prototypes into the updated
        feature space. The transport itself is discarded after application."""
        z_old_raw = self._lrpt_pre_features
        z_new_raw, z_new_targets = self._extract_current_task_features(
            data_manager, return_targets=True
        )
        z_old_targets = self._lrpt_pre_targets
        self._lrpt_pre_features = None
        self._lrpt_pre_targets = None
        if self.args.get("sa_raw_prototypes", False):
            z_old = z_old_raw.float()
            z_new = z_new_raw.float()
        else:
            z_old = F.normalize(z_old_raw, p=2, dim=1)
            z_new = F.normalize(z_new_raw, p=2, dim=1)
        if self._lrpt_fit_target == "classmean":
            fit_old = self._class_mean_prototypes(z_old, z_old_targets)
            fit_new = self._class_mean_prototypes(z_new, z_new_targets)
            logging.info(
                "[SharedA-SDLoRA] LRPT class-mean fit: %d current classes",
                fit_old.shape[0],
            )
        else:
            fit_old, fit_new = z_old, z_new

        if self._lrpt_basis != "generic":
            u, v, bias = self._fit_lora_aware_transport(
                data_manager, fit_old, fit_new
            )
            rel_err, _ = transport_prediction_error(
                fit_old, fit_new, u, v, bias=bias
            )
        elif self._lrpt_bias:
            if self._lrpt_dual:
                u_s, v_s, b_s = fit_affine_low_rank_transport(
                    z_old,
                    z_new,
                    rank=self._lrpt_rank,
                    reg=self._lrpt_reg,
                )
                u_c, v_c, b_c = fit_affine_low_rank_transport(
                    fit_old,
                    fit_new,
                    rank=self._lrpt_rank,
                    reg=self._lrpt_reg,
                )
                w = max(self._lrpt_class_weight, 0.0)
                u = torch.cat([u_s, math.sqrt(w) * u_c], dim=1)
                v = torch.cat([v_s, math.sqrt(w) * v_c], dim=1)
                bias = b_s + w * b_c
                rel_err, _ = transport_prediction_error(
                    fit_old, fit_new, u, v, bias=bias
                )
                logging.info(
                    "[SharedA-SDLoRA] LRPT dual-space: sample_rank=%d class_rank=%d class_weight=%.2f",
                    u_s.shape[1],
                    u_c.shape[1],
                    w,
                )
            else:
                u, v, bias = fit_affine_low_rank_transport(
                    fit_old,
                    fit_new,
                    rank=self._lrpt_rank,
                    reg=self._lrpt_reg,
                )
                rel_err, _ = transport_prediction_error(
                    fit_old, fit_new, u, v, bias=bias
                )
        else:
            if self._lrpt_dual:
                raise ValueError("lrpt_dual requires lrpt_bias=True")
            u, v, bias = (
                *fit_low_rank_transport(
                    fit_old,
                    fit_new,
                    rank=self._lrpt_rank,
                    reg=self._lrpt_reg,
                ),
                None,
            )
            rel_err, _ = transport_prediction_error(fit_old, fit_new, u, v)
        if self._lrpt_diagnostics:
            diag_norm = fit_rank_residuals(
                z_old, z_new, (10, 16, 32, 768), reg=self._lrpt_reg
            )
            diag_raw = fit_rank_residuals(
                z_old_raw.to(torch.float32),
                z_new_raw.to(torch.float32),
                (10, 16, 32, 768),
                reg=self._lrpt_reg,
            )
            diag_affine_norm = fit_affine_rank_residuals(
                z_old, z_new, (10, 16, 32, 768), reg=self._lrpt_reg
            )
            diag_affine_raw = fit_affine_rank_residuals(
                z_old_raw.to(torch.float32),
                z_new_raw.to(torch.float32),
                (10, 16, 32, 768),
                reg=self._lrpt_reg,
            )
            classmean_affine = {}
            if (
                z_old_targets is not None
                and z_new_targets is not None
                and torch.equal(z_old_targets, z_new_targets)
            ):
                old_means = self._class_mean_prototypes(
                    z_old, z_old_targets
                )
                new_means = self._class_mean_prototypes(
                    z_new, z_new_targets
                )
                classmean_affine = fit_affine_rank_residuals(
                    old_means,
                    new_means,
                    (10, 16, 32, 768),
                    reg=self._lrpt_reg,
                )
            bias_rel = bias_relative_error(z_old, z_new)
            logging.info(
                "[SharedA-SDLoRA] LRPT-DIAG task %d norm=%s raw=%s affine_norm=%s affine_raw=%s classmean_affine=%s bias_rel=%.4f",
                self._cur_task,
                {k: round(v, 4) for k, v in diag_norm.items()},
                {k: round(v, 4) for k, v in diag_raw.items()},
                {k: round(v, 4) for k, v in diag_affine_norm.items()},
                {k: round(v, 4) for k, v in diag_affine_raw.items()},
                {k: round(v, 4) for k, v in classmean_affine.items()},
                bias_rel,
            )
        logging.info(
            "[SharedA-SDLoRA] LRPT task %d: rank=%d reg=%.2e relative_drift_error=%.4f",
            self._cur_task,
            self._lrpt_rank,
            self._lrpt_reg,
            rel_err,
        )

        path = os.path.join(self.args["filepath"], PROTOTYPES_FILENAME)
        if not os.path.exists(path):
            logging.warning(
                "[SharedA-SDLoRA] LRPT skipped: no stored prototypes at %s",
                path,
            )
            return
        old = torch.load(path, map_location="cpu", weights_only=True)
        if self._lrpt_adaptive and bias is not None:
            updated = apply_transport_adaptive(old, u, v, bias=bias)
        else:
            updated = apply_transport(
                old, u, v, bias=bias, damping=self._lrpt_damping
            )
        torch.save(updated, path)
        if bias is not None:
            logging.info(
                "[SharedA-SDLoRA] LRPT bias norm=%.4f",
                float(torch.linalg.norm(bias)),
            )
        logging.info(
            "[SharedA-SDLoRA] LRPT moved %d old prototypes to updated feature space",
            len(updated),
        )

    def _class_mean_prototypes(self, features, targets):
        """Per-class mean of L2-normalized features, then L2-normalized, which
        matches the stored prototype representation used by PrototypeCosineHead."""
        class_ids = torch.unique(targets)
        means = []
        for class_id in class_ids:
            mask = targets == class_id
            mean = features[mask].mean(dim=0)
            means.append(F.normalize(mean, p=2, dim=0))
        return torch.stack(means)

    def _compute_prototypes(self, data_manager, raw_network):
        """Compute L2-normalized per-class mean prototypes for the classes of
        the current task, using only that task's training data at training time.
        Old prototypes are loaded from disk and merged (no old data re-use)."""
        cur_classes = np.arange(self._known_classes, self._total_classes)
        dataset = data_manager.get_dataset(
            cur_classes, source="train", mode="test"
        )
        raw_network.eval()
        per_class = {c: [] for c in cur_classes.tolist()}
        all_features = []
        all_targets = []
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
                feats = raw_network.backbone(inputs)
                if not self.args.get("sa_raw_prototypes", False):
                    feats = F.normalize(feats, p=2, dim=1)
                all_features.append(feats.cpu())
                all_targets.append(targets)
                for f, target in zip(feats.cpu(), targets):
                    per_class[int(target.item())].append(f)
        self._last_proto_features = torch.cat(all_features)
        self._last_proto_targets = torch.cat(all_targets)
        self._last_proto_task = self._cur_task
        k_prototypes = int(self.args.get("sa_k_prototypes", 1))
        if k_prototypes <= 1:
            prototypes = {
                c: F.normalize(torch.stack(per_class[c]).mean(dim=0), p=2, dim=0)
                for c in cur_classes.tolist()
            }
        else:
            prototypes = {
                c: self._cluster_prototypes(per_class[c], k_prototypes)
                for c in cur_classes.tolist()
            }
        path = os.path.join(self.args["filepath"], PROTOTYPES_FILENAME)
        if os.path.exists(path):
            old = torch.load(path, map_location="cpu", weights_only=True)
            for class_id, vector in old.items():
                prototypes.setdefault(int(class_id), vector)
        torch.save(prototypes, path)
        return prototypes

    @staticmethod
    def _cluster_prototypes(features, k, iterations=20):
        feats = torch.stack(features)
        if len(feats) <= k:
            mean = F.normalize(feats.mean(dim=0), p=2, dim=0)
            return [mean.clone() for _ in range(k)]
        centers = [feats[0].clone()]
        for _ in range(1, k):
            dists = (feats - centers[-1]).pow(2).sum(dim=1)
            centers.append(feats[dists.argmax()].clone())
        centers = torch.stack(centers)
        for _ in range(iterations):
            dists = torch.stack(
                [(feats - center).pow(2).sum(dim=1) for center in centers]
            )
            assignments = dists.argmin(dim=0)
            new_centers = []
            for j in range(k):
                mask = assignments == j
                if mask.sum() > 0:
                    new_centers.append(feats[mask].mean(dim=0))
                else:
                    new_centers.append(centers[j].clone())
            centers = torch.stack(new_centers)
        return [
            F.normalize(center, p=2, dim=0).detach().clone()
            for center in centers
        ]

    def _rebuild_eval_backbone(self):
        """Rebuild the backbone from saved artifacts so each task counts once."""
        self._unwrap_network()
        self._network.backbone = self.update_network(
            index=False, task_index=self._cur_task + 1
        )
        self._network.to(self._device)

    def _final_head_tune(self, data_manager, raw_network, epochs, lr):
        """Freeze the backbone and fine-tune only the global classifier."""
        raw_network.eval()
        for param in raw_network.backbone.parameters():
            param.requires_grad_(False)

        dataset = data_manager.get_dataset(
            np.arange(0, data_manager.nb_classes),
            source="train",
            mode="train",
        )
        loader = DataLoader(
            dataset,
            batch_size=min(int(self.args["batch_size"]), 16),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        features, targets = [], []
        torch.cuda.empty_cache()
        with torch.no_grad():
            for _, inputs, batch_targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                features.append(raw_network.backbone(inputs).detach().cpu())
                targets.append(batch_targets)
        feature_matrix = torch.cat(features)
        target_vector = torch.cat(targets)
        del features, targets
        torch.cuda.empty_cache()

        raw_network.fc.train()
        optimizer = optim.SGD(
            raw_network.fc.parameters(), lr=lr, momentum=0.9
        )
        for epoch in range(epochs):
            permutation = torch.randperm(feature_matrix.shape[0])
            epoch_loss = 0.0
            steps = 0
            for start in range(0, feature_matrix.shape[0], self.args["batch_size"]):
                indices = permutation[start : start + self.args["batch_size"]]
                batch_x = feature_matrix[indices].to(self._device)
                batch_y = target_vector[indices].to(self._device)
                logits = raw_network.fc(batch_x)["logits"]
                loss = F.cross_entropy(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                steps += 1
            logging.info(
                "[SharedA-SDLoRA] final head tune epoch %d/%d loss=%.4f",
                epoch + 1,
                epochs,
                epoch_loss / max(steps, 1),
            )
