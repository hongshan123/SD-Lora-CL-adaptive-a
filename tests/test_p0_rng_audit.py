"""P0 determinism audit tests from p0_rng_audit_fix_guide_sd.md.

Covers: fresh-run guard, same-run task1 loading, filepath lock, strict
deterministic backend, RNG/tensor invariance, four-rank sync, single-pass
three-head eval, and canonical tensor hashing.
"""

import json
import multiprocessing
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import (
    SA_STATE_FILENAME,
    SharedALoRA_ViT_timm,
)
from models.sa_sdlora import (
    evaluate_dual_head_once,
    dual_head_logits,
)
from trainer import _set_deterministic_backend
from utils.canonical_hash import (
    canonical_tensor_hash,
    compare_named_tensors,
    hash_named_tensors,
    model_tensor_map,
)
from utils.run_guard import acquire_run_guard
from utils.rng_utils import (
    rng_preserving,
    rng_state_hash,
)


class _TinyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x):
        return self.qkv(x)


class _TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _TinyAttention(dim)

    def forward(self, x):
        return self.attn(x)


class _TinyViT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dim)])
        self.head = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def test_task0_rejects_existing_sa_state(tmp_path):
    torch.save({"version": -1}, tmp_path / SA_STATE_FILENAME)
    with pytest.raises(FileExistsError, match="fresh-run guard"):
        SharedALoRA_ViT_timm(
            _TinyViT(6),
            r=2,
            filepath=str(tmp_path),
            cur_task_index=0,
        )


def test_task1_loads_same_run_task0_state(tmp_path):
    run = tmp_path / "run"
    model = SharedALoRA_ViT_timm(
        _TinyViT(6),
        r=2,
        filepath=str(run),
        cur_task_index=0,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        train_a_all_tasks=True,
    )
    with torch.no_grad():
        for w in model.w_Bs:
            w.weight.copy_(torch.randn_like(w.weight))
    model.save_lora_parameters(str(run), task_id=0)

    reloaded = SharedALoRA_ViT_timm(
        _TinyViT(6),
        r=2,
        filepath=str(run),
        cur_task_index=1,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        train_a_all_tasks=True,
    )
    assert len(reloaded.aggregate_up) == 2
    assert all(torch.all(w.weight == 0) for w in reloaded.w_Bs)


def _guard_child(path, queue):
    try:
        lock = acquire_run_guard(str(path))
        queue.put(("ok", lock.fileno()))
    except Exception as exc:  # noqa: BLE001 - report any failure mode
        queue.put(("err", type(exc).__name__))


def test_second_process_fails_same_filepath(tmp_path):
    target = tmp_path / "run"
    lock = acquire_run_guard(str(target), config_bytes=b"{}", run_id="parent")
    try:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        proc = context.Process(
            target=_guard_child, args=(str(target), queue)
        )
        proc.start()
        proc.join(timeout=30)
        kind, detail = queue.get(timeout=10)
        assert kind == "err"
        assert detail in ("RuntimeError", "FileExistsError")
    finally:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def test_run_manifest_records_metadata(tmp_path):
    target = tmp_path / "run"
    lock = acquire_run_guard(
        str(target),
        config_bytes=b'{"seed": 1}',
        command=["torchrun", "main.py"],
        run_id="run-abc",
    )
    import fcntl

    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()
    manifest = json.loads(
        (target / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_id"] == "run-abc"
    assert manifest["command"] == ["torchrun", "main.py"]
    assert manifest["config_sha256"] == (
        "a4bbadc1c78966d28872dbe60ff40cb9c996f884e228ed4a08eb7d6c3a24f4b8"
    )


def test_strict_deterministic_backend_configuration():
    previous = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }
    try:
        _set_deterministic_backend({"sa_deterministic_training": True})
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.flash_sdp_enabled() is False
        assert torch.backends.cuda.mem_efficient_sdp_enabled() is False
        assert torch.backends.cuda.math_sdp_enabled() is True
    finally:
        torch.use_deterministic_algorithms(
            previous["deterministic"], warn_only=False
        )
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = previous["tf32_matmul"]
        torch.backends.cudnn.allow_tf32 = previous["tf32_cudnn"]
        torch.backends.cuda.enable_flash_sdp(previous["flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(
            previous["mem_efficient"]
        )
        torch.backends.cuda.enable_math_sdp(previous["math"])


def test_rng_hash_stable_under_rng_preserving():
    before = rng_state_hash()
    with rng_preserving():
        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(321)
    after = rng_state_hash()
    assert before == after

    torch.manual_seed(999)
    changed = rng_state_hash()
    assert changed != before


def test_model_tensor_hash_invariant_under_no_grad_forward():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(6, 6), nn.ReLU())
    before_map = model_tensor_map(model)
    before_hash = hash_named_tensors(before_map)
    with torch.no_grad():
        model(torch.randn(3, 6))
    after_map = model_tensor_map(model)
    assert compare_named_tensors(before_map, after_map) is None
    assert before_hash == hash_named_tensors(after_map)


class _FakeHead(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, features):
        return {"logits": self.scale * features}


class _FakeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.fc = _FakeHead(1.0)
        self.prototype_head = _FakeHead(2.0)
        self.dual_lambda = 0.5
        self.tau_fc = 1.0
        self.tau_proto = 2.0


class _CountingDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        self.calls = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.calls += 1
        return index, self.data[index], torch.tensor(index % 4)


def test_three_head_eval_traverses_loader_once():
    dataset = _CountingDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    preds, y_true, _ = evaluate_dual_head_once(
        _FakeNetwork(), loader, topk=2
    )
    assert dataset.calls == len(dataset)
    assert len(y_true[0]) == 4
    assert preds["fc"][0].shape == (4, 2)
    assert preds["proto"][0].shape == (4, 2)
    assert preds["fused"][0].shape == (4, 2)


def test_dual_head_formula_single_pass_smoke():
    network = _FakeNetwork()
    features = torch.randn(3, 5)
    fc, proto, fused = dual_head_logits(network, features)
    assert torch.allclose(fc, features)
    assert torch.allclose(proto, features)
    assert torch.allclose(fused, 0.5 * features + 0.5 * features)


def test_canonical_hash_stable_and_sensitive():
    tensor = torch.randn(4, 4)
    clone = tensor.clone()
    assert canonical_tensor_hash("x", tensor) == canonical_tensor_hash(
        "x", clone
    )
    changed = tensor.clone()
    changed[0, 0] = changed[0, 0] + 1e-6
    assert canonical_tensor_hash("x", tensor) != canonical_tensor_hash(
        "x", changed
    )
    assert canonical_tensor_hash("x", tensor) != canonical_tensor_hash(
        "y", tensor
    )


def test_four_rank_prototype_and_dual_head_sync():
    script = Path(__file__).resolve().parent / "ddp_p0_rank_sync.py"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [sys.executable, str(script)],
        check=True,
        timeout=120,
        env=env,
    )
