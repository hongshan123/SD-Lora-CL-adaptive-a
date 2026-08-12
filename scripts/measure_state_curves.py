#!/usr/bin/env python3
"""Persistent-state and per-task-time curves at T=5/10/20 (frozen protocol).

For the Live-A Dual-B method and its baselines (SD-LoRA, EXP-009, rank-1
SD-LoRA+Dual-B) on ImageNet-R seed1995, reports:
  * trainable/persistent LoRA + head parameter count;
  * minimal persistent disk bytes (state + per-task B when applicable +
    final FC + prototypes + dual head);
  * per-task wall-clock time from log timestamps.

Usage:
  python scripts/measure_state_curves.py
"""

import os
import re
from datetime import datetime

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ARTIFACTS = {
    "full": {
        5: "INR_P5_FULL_SEED1995_T5_NCCL",
        10: "INR_P1_LIVEA_DUALB_SEED1995_NCCL",
        20: "INR_P5_FULL_SEED1995_T20_NCCL",
    },
    "sdlora": {
        5: "INR_P5_SDLORA_SEED1995_T5_NCCL",
        10: "INR_P1_SDLORA_SEED1995_NCCL",
        20: "INR_P5_SDLORA_SEED1995_T20_NCCL",
    },
    "exp009": {
        5: "INR_P5_EXP009_SEED1995_T5_NCCL",
        10: "INR_P1_EXP009_SEED1995_NCCL",
        20: "INR_P5_EXP009_SEED1995_T20_NCCL",
    },
    "rank1": {
        5: "INR_P5_RANK1_SEED1995_T5_NCCL",
        10: "INR_P1_RANK1_SDLORA_SEED1995_NCCL",
        20: "INR_P5_RANK1_SEED1995_T20_NCCL",
    },
}

LOGS = {
    "full": {5: "p5_full_t5_inr_nccl.log", 10: "inr_p1_livea_dual_b_seed1995_nccl.log",
             20: "p5_full_t20_inr_nccl.log"},
    "sdlora": {5: "p5_sdlora_t5_inr_nccl.log", 10: "inr_p1_sdlora_seed1995_nccl.log",
               20: "p5_sdlora_t20_inr_nccl.log"},
    "exp009": {5: "p5_exp009_t5_inr_nccl.log", 10: "inr_p1_exp009_seed1995_nccl.log",
               20: "p5_exp009_t20_inr_nccl.log"},
    "rank1": {5: "p5_rank1_t5_inr_nccl.log", 10: "inr_p1_rank1_sdlora_seed1995_nccl.log",
              20: "p5_rank1_t20_inr_nccl.log"},
}


def _count_tensors(values):
    if isinstance(values, dict):
        values = list(values.values())
    total = 0
    for value in values:
        if torch.is_tensor(value):
            total += value.numel()
        elif isinstance(value, (list, tuple)):
            total += _count_tensors(value)
        elif hasattr(value, "state_dict"):
            total += _count_tensors(value.state_dict())
    return total


def count_artifact(directory, num_tasks):
    state_path = os.path.join(directory, "sa_state.pt")
    lora_params = 0
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        version = int(state.get("version", -1))
        if version == 4:
            lora_params = _count_tensors(
                state.get("shared_a", [])
            ) + _count_tensors(state.get("aggregate_up", []))
        elif version in (2, 3):
            lora_params = (
                _count_tensors(state.get("canonical_down", []))
                + _count_tensors(state.get("cumulative_up", []))
                + _count_tensors(state.get("triangular_r", []))
            )
        else:
            lora_params = _count_tensors(
                state.get("shared_a", [])
            ) + _count_tensors(state.get("scales", {}))
    else:
        for task in range(num_tasks):
            path = os.path.join(directory, "lora_w_a_{}.pt".format(task))
            if os.path.exists(path):
                lora_params += _count_tensors(
                    torch.load(path, map_location="cpu", weights_only=False)
                )
    b_files = 0
    for task in range(num_tasks):
        path = os.path.join(directory, "sa_lora_w_b_{}.pt".format(task))
        if os.path.exists(path):
            b_files += _count_tensors(
                torch.load(path, map_location="cpu", weights_only=True)
            )
        path = os.path.join(directory, "lora_w_b_{}.pt".format(task))
        if os.path.exists(path):
            b_files += _count_tensors(
                torch.load(path, map_location="cpu", weights_only=False)
            )
    lora_params += b_files

    proto_path = os.path.join(directory, "sa_prototypes.pt")
    proto_params = 0
    if os.path.exists(proto_path):
        proto_params = _count_tensors(
            torch.load(proto_path, map_location="cpu", weights_only=True)
        )
    task_id = num_tasks - 1
    weight = torch.load(
        os.path.join(directory, "CLs_weight{}.pt".format(task_id)),
        map_location="cpu",
        weights_only=True,
    )
    bias = torch.load(
        os.path.join(directory, "CLs_bias{}.pt".format(task_id)),
        map_location="cpu",
        weights_only=True,
    )
    head_params = weight.numel() + bias.numel()
    return lora_params, proto_params, head_params


def disk_bytes(directory, num_tasks):
    task_id = num_tasks - 1
    names = []
    if os.path.exists(os.path.join(directory, "sa_state.pt")):
        names.append("sa_state.pt")
    for task in range(num_tasks):
        for stem in ("sa_lora_w_b_{}.pt", "lora_w_a_{}.pt", "lora_w_b_{}.pt"):
            path = os.path.join(directory, stem.format(task))
            if os.path.exists(path):
                names.append(os.path.basename(path))
    names += [
        "CLs_weight{}.pt".format(task_id),
        "CLs_bias{}.pt".format(task_id),
    ]
    for extra in ("sa_prototypes.pt", "sa_dual_head.pt"):
        if os.path.exists(os.path.join(directory, extra)):
            names.append(extra)
    total = 0
    for name in names:
        total += os.path.getsize(os.path.join(directory, name))
    return total


def per_task_times(log_path, num_tasks):
    text = open(log_path, errors="ignore").read()
    starts = re.findall(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[sdlora\.py\] => Learning on ",
        text,
    )
    ends = re.findall(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[sdlora\.py\] => Task \d+, Epoch 20/20",
        text,
    )
    fmt = "%Y-%m-%d %H:%M:%S"
    times = []
    for task in range(num_tasks):
        if task >= len(starts):
            times.append(float("nan"))
            continue
        start = datetime.strptime(starts[task], fmt)
        if task + 1 < len(starts):
            end = datetime.strptime(starts[task + 1], fmt)
        elif task < len(ends):
            end = datetime.strptime(ends[task], fmt)
        else:
            end = start
        times.append((end - start).total_seconds())
    return times


def main():
    print("{:<7} {:>3} {:>10} {:>10} {:>10} {:>12} {:>10}".format(
        "method", "T", "lora", "proto", "head", "disk_B", "mean_s"
    ))
    per_task_t10 = {}
    for method in ("full", "sdlora", "exp009", "rank1"):
        for tasks in (5, 10, 20):
            artifact = os.path.join(ROOT, ARTIFACTS[method][tasks])
            lora, proto, head = count_artifact(artifact, tasks)
            bytes_ = disk_bytes(artifact, tasks)
            times = per_task_times(
                os.path.join(ROOT, LOGS[method][tasks]), tasks
            )
            if tasks == 10:
                per_task_t10[method] = times
            mean_time = sum(times) / max(len(times), 1)
            print("{:<7} {:>3} {:>10,} {:>10,} {:>10,} {:>12,} {:>10.1f}".format(
                method, tasks, lora, proto, head, bytes_, mean_time
            ))

    print("")
    print("Per-task wall time (s) at T=10 (task 0..9):")
    print("{:<7} {}".format("method", " ".join(
        "{:>7}".format("t{}".format(t)) for t in range(10)
    )))
    for method, times in per_task_t10.items():
        print("{:<7} {}".format(method, " ".join(
            "{:>7.1f}".format(t) for t in times
        )))


if __name__ == "__main__":
    main()
