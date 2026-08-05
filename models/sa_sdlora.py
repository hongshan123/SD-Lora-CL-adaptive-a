import logging
import os

import numpy as np
import timm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from backbone.lrpt import (
    apply_transport,
    bias_relative_error,
    fit_low_rank_transport,
    fit_rank_residuals,
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
        self._lrpt_diagnostics = bool(args.get("lrpt_diagnostics", True))
        self._lrpt_pre_features = None

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
            self._lrpt_pre_features = self._extract_current_task_features(
                data_manager,
                task_index=self._cur_task + 1,
            )
            logging.info(
                "[SharedA-SDLoRA] LRPT pre-update features captured for task %d (%d samples)",
                self._cur_task + 1,
                self._lrpt_pre_features.shape[0],
            )
        super().incremental_train(data_manager)
        if self._is_main_process():
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

    def _extract_current_task_features(
        self, data_manager, task_index=None, normalize=False
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
        with torch.no_grad():
            for _, inputs, _ in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                feats = raw_network.backbone(inputs)
                if normalize:
                    feats = F.normalize(feats, p=2, dim=1)
                features.append(feats.detach().cpu())
        return torch.cat(features, dim=0)

    def _apply_lrpt_to_old_prototypes(self, data_manager):
        """Fit the low-rank transport from the paired current-task features and
        recursively move all previously stored prototypes into the updated
        feature space. The transport itself is discarded after application."""
        z_old_raw = self._lrpt_pre_features
        z_new_raw = self._extract_current_task_features(data_manager)
        self._lrpt_pre_features = None
        z_old = F.normalize(z_old_raw, p=2, dim=1)
        z_new = F.normalize(z_new_raw, p=2, dim=1)

        u, v = fit_low_rank_transport(
            z_old,
            z_new,
            rank=self._lrpt_rank,
            reg=self._lrpt_reg,
        )
        rel_err, _ = transport_prediction_error(z_old, z_new, u, v)
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
            bias_rel = bias_relative_error(z_old, z_new)
            logging.info(
                "[SharedA-SDLoRA] LRPT-DIAG task %d norm=%s raw=%s bias_rel=%.4f",
                self._cur_task,
                {k: round(v, 4) for k, v in diag_norm.items()},
                {k: round(v, 4) for k, v in diag_raw.items()},
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
        updated = apply_transport(old, u, v)
        torch.save(updated, path)
        logging.info(
            "[SharedA-SDLoRA] LRPT moved %d old prototypes to updated feature space",
            len(updated),
        )

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
