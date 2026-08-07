import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

import timm
from backbone.lora import LoRA_ViT_timm
import torch.distributed as dist

import os

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

    def _is_distributed(self):
        return self.args.get("distributed", False)

    def _is_main_process(self):
        return self.args.get("rank", 0) == 0

    def _barrier(self):
        if self._is_distributed() and dist.is_initialized():
            dist.barrier()

    def _unwrap_network(self):
        if isinstance(self._network, DDP) or isinstance(self._network, nn.DataParallel):
            self._network = self._network.module
        return self._network

    def _wrap_ddp(self):
        if self._is_distributed():
            local_rank = self.args["local_rank"]
            self._network = DDP(
                self._network,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )

    def _raw_network(self):
        if isinstance(self._network, DDP) or isinstance(self._network, nn.DataParallel):
            return self._network.module
        return self._network

    def _additional_training_losses(
        self, inputs=None, targets=None, features=None
    ):
        """Optional learner-specific loss terms for incremental tasks."""
        return {}

    def _sync_sum(self, value):
        tensor = torch.tensor(float(value), device=self._device)
        if self._is_distributed():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        if self._is_main_process():
            logging.info(
                "Learning on {}-{}".format(self._known_classes, self._total_classes)
            )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=self.args["world_size"],
                rank=self.args["rank"],
                shuffle=True,
                drop_last=False,
            )
            if self._is_distributed()
            else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        if len(self._multiple_gpus) > 1 and not self._is_distributed():
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1 and not self._is_distributed():
            self._network = self._network.module

    def update_network(self, index=True):
        # if use VIT-B-16
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)

        # if use DINO
        # model = timm.create_model('vit_base_patch16_224_dino', pretrained=True, num_classes=0)

        # SD-LoRA-RR
        '''
        if self._cur_task >=4 and self._cur_task <8:
            rank = 8 #8
        elif self._cur_task >=8:
            rank = 6 #6
        # elif self._cur_task >=8:
        #     rank = 4
        else:
            rank = 10
        '''
        rank = int(self.args.get("lora_rank", 10))
        if rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self._is_main_process():
            logging.info(
                "[SDLoRA] task=%d rank=%d task_lora_params=%d",
                self._cur_task,
                rank,
                12 * 2 * 2 * 768 * rank,
            )
        model = LoRA_ViT_timm(vit_model=model.eval(), r=rank, num_classes=10, index=index, increment= self.args['increment'], filepath=self.args['filepath'], 
        cur_task_index= self._cur_task)
        model.out_dim = 768
        return model

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            self._wrap_ddp()
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                # weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)

        else:
            self._unwrap_network()
            self._network.backbone = self.update_network(index=False)
            if len(self._multiple_gpus) > 1 and not self._is_distributed():
                self._network = nn.DataParallel(self._network, self._multiple_gpus)       
            self._network.to(self._device) 
            self._wrap_ddp()

            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        save_lora_name = self.args['filepath']

        raw_network = self._raw_network()
        if self._is_main_process():
            raw_network.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            raw_network.save_fc(save_lora_name, self._cur_task)

        self._barrier()
        self._unwrap_network()

    def get_optimizer(self):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                # lr=self.init_lr, 
                self.args["lrate"],
                # weight_decay=self.weight_decay
                betas=(0.9, 0.999)
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.args['min_lr'])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler



    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(
            range(self.args["init_epoch"]),
            disable=not self._is_main_process(),
        )
        for _, epoch in enumerate(prog_bar):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            self._network.train()
            losses = 0.0
            extra_loss_sums = {}
            correct, total = 0, 0
            num_batches = 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network(inputs)
                logits = out["logits"]
                features = out.get("features", None)
                loss = F.cross_entropy(logits, targets)
                extra_losses = self._additional_training_losses(
                    inputs, targets, features=features
                )
                for name, extra_loss in extra_losses.items():
                    if extra_loss.ndim != 0:
                        raise ValueError("additional training losses must be scalar")
                    loss = loss + extra_loss
                    extra_loss_sums[name] = (
                        extra_loss_sums.get(name, 0.0) + extra_loss.detach().item()
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                num_batches += 1

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).sum()
                total += len(targets)

            scheduler.step()
            loss_sum = self._sync_sum(losses)
            batch_sum = self._sync_sum(num_batches)
            correct_sum = self._sync_sum(correct.item())
            total_sum = self._sync_sum(total)
            train_acc = round(correct_sum * 100 / max(total_sum, 1), 2)
            extra_loss_info = ""
            for name, value in extra_loss_sums.items():
                value_sum = self._sync_sum(value)
                extra_loss_info += ", {} {:.4f}".format(
                    name, value_sum / max(batch_sum, 1)
                )

            test_acc = None
            if epoch % 5 == 0:
                self._barrier()
                if self._is_main_process():
                    test_acc = self._compute_accuracy(self._raw_network(), test_loader)
                self._barrier()

            if test_acc is not None:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}{}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    loss_sum / max(batch_sum, 1),
                    train_acc,
                    test_acc,
                    extra_loss_info,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}{}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    loss_sum / max(batch_sum, 1),
                    train_acc,
                    extra_loss_info,
                )

            if self._is_main_process():
                prog_bar.set_description(info)

        if self._is_main_process():
            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]), disable=not self._is_main_process())
        for _, epoch in enumerate(prog_bar):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            self._network.train()
            losses = 0.0
            extra_loss_sums = {}
            correct, total = 0, 0
            num_batches = 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                if i == 0:
                    raw_network = self._raw_network()
                    backbone = raw_network.backbone
                    tokens = backbone.lora_vit.patch_embed(inputs)
                    tokens = backbone.lora_vit.pos_drop(tokens)
                    live_diag = (
                        backbone.live_a_gradient_diagnostics(tokens)
                    )
                    if live_diag is not None and live_diag["historical_dL_dA"] > 0:
                        logging.info(
                            "[LiveA-SDLoRA] task %d first-batch diag: "
                            "hist_dL_dA=%.4e cur_dL_dA=%.4e ratio=%.3f "
                            "G=%.4e A=%.4e B=%.4e scale=%.4f",
                            self._cur_task,
                            live_diag["historical_dL_dA"],
                            live_diag["current_dL_dA"],
                            live_diag["ratio_hist_cur"],
                            live_diag["mean_G_norm"],
                            live_diag["mean_A_norm"],
                            live_diag["mean_B_norm"],
                            live_diag["scale"],
                        )
                # logits = self._network(inputs)["logits"]
                logits, ortho_loss = self._network(inputs, ortho_loss=True)
                features = logits.get("features", None)
                logits = logits['logits'] 
                

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                )
                # print('@@@@@@@@@@@@@@loss2', loss_clf, torch.mean(ortho_loss))

                loss = loss_clf
                extra_losses = self._additional_training_losses(
                    inputs, targets, features=features
                )
                for name, extra_loss in extra_losses.items():
                    if extra_loss.ndim != 0:
                        raise ValueError("additional training losses must be scalar")
                    loss = loss + extra_loss
                    extra_loss_sums[name] = (
                        extra_loss_sums.get(name, 0.0) + extra_loss.detach().item()
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                num_batches += 1

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).sum()
                total += len(targets)

            scheduler.step()
            loss_sum = self._sync_sum(losses)
            batch_sum = self._sync_sum(num_batches)
            correct_sum = self._sync_sum(correct.item())
            total_sum = self._sync_sum(total)
            train_acc = round(correct_sum * 100 / max(total_sum, 1), 2)
            extra_loss_info = ""
            for name, value in extra_loss_sums.items():
                value_sum = self._sync_sum(value)
                extra_loss_info += ", {} {:.4f}".format(
                    name, value_sum / max(batch_sum, 1)
                )

            test_acc = None
            if epoch % 5 == 0:
                self._barrier()
                if self._is_main_process():
                    test_acc = self._compute_accuracy(self._raw_network(), test_loader)
                self._barrier()

            if test_acc is not None:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}{}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    loss_sum / max(batch_sum, 1),
                    train_acc,
                    test_acc,
                    extra_loss_info,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}{}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    loss_sum / max(batch_sum, 1),
                    train_acc,
                    extra_loss_info,
                )
            if self._is_main_process():
                prog_bar.set_description(info)
        if self._is_main_process():
            logging.info(info)
