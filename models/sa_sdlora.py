import logging
import math
import os

import numpy as np
import timm
import torch
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
from backbone.sa_lora import SharedALoRA_ViT_timm
from utils.inc_net import SimpleCosineIncrementalNet, SharedAPrototypeNet
from models.sdlora import Learner as SDLoraLearner


num_workers = 8
PROTOTYPES_FILENAME = "sa_prototypes.pt"


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
        )
        model.out_dim = 768
        return model

    def incremental_train(self, data_manager):
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
        if self._is_main_process():
            if self._cur_task > 0:
                with torch.no_grad():
                    backbone = self._raw_network().backbone
                    if backbone.cumulative_state:
                        gauge_residual = backbone.cumulative_gauge_residual()
                        logging.info(
                            "[SharedA-SDLoRA] cumulative gauge task %d: "
                            "relative_projection_residual=%.6f",
                            self._cur_task,
                            gauge_residual,
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
                if self._lrpt_enabled and self._lrpt_pre_features is not None:
                    self._apply_lrpt_to_old_prototypes(data_manager)
                prototypes = self._compute_prototypes(data_manager, raw_network)
                raw_network.set_prototypes(prototypes)
                logging.info(
                    "[SharedA-SDLoRA] prototype classifier active for %d classes",
                    len(prototypes),
                )
                raw_network.backbone.cleanup_per_task_files(self.args["filepath"])

    def _additional_training_losses(
        self, inputs=None, targets=None, features=None
    ):
        losses = {}
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
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        raw_network = self._raw_network()
        raw_network.eval()
        features = []
        targets = []
        with torch.no_grad():
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
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        raw_network.eval()
        sums = {c: None for c in cur_classes.tolist()}
        counts = {c: 0 for c in cur_classes.tolist()}
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                feats = raw_network.backbone(inputs)
                if not self.args.get("sa_raw_prototypes", False):
                    feats = F.normalize(feats, p=2, dim=1)
                for f, target in zip(feats.cpu(), targets):
                    c = int(target.item())
                    if sums[c] is None:
                        sums[c] = f.clone()
                    else:
                        sums[c] = sums[c] + f
                    counts[c] += 1
        prototypes = {c: F.normalize(sums[c], p=2, dim=0) for c in cur_classes.tolist()}
        path = os.path.join(self.args["filepath"], PROTOTYPES_FILENAME)
        if os.path.exists(path):
            old = torch.load(path, map_location="cpu", weights_only=True)
            for class_id, vector in old.items():
                prototypes.setdefault(int(class_id), vector)
        torch.save(prototypes, path)
        return prototypes

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
