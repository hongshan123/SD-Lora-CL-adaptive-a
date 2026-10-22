import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.cuo_lowrank import (
    advance_projected_cuo,
    row_orthonormal_projection,
    solve_projected_cuo,
)
from backbone.lora import ParameterWrapper
from backbone.sa_lora import (
    SA_MERGED_FILENAME,
    SA_STATE_FILENAME,
    SA_STATE_VERSION_CUO,
    SharedALoRA_ViT_timm,
    _CUOLowRankQKV,
    combine_cuo_statistics,
    cuo_state_scalar_counts,
)
from models.sa_sdlora import Learner as SharedALearner
from models.sa_sdlora import validate_cuo_lowrank_config


def make_cuo_wrapper(dim=8, rank=3):
    qkv = nn.Linear(dim, 3 * dim, bias=False)
    a_q = nn.Linear(dim, rank, bias=False)
    a_v = nn.Linear(dim, rank, bias=False)
    b_q = nn.Linear(rank, dim, bias=False)
    b_v = nn.Linear(rank, dim, bias=False)
    projection_q = torch.randn(rank, dim)
    projection_v = torch.randn(rank, dim)
    unified_up_q = torch.randn(dim, rank)
    unified_up_v = torch.randn(dim, rank)
    scaling = nn.ModuleList(
        [ParameterWrapper(nn.Parameter(torch.tensor([0.7])))]
    )
    return _CUOLowRankQKV(
        qkv,
        a_q,
        a_v,
        b_q,
        b_v,
        projection_q,
        unified_up_q,
        projection_v,
        unified_up_v,
        scaling,
        layer_index=0,
    )


class _CUOTaskEndAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)


class _CUOTaskEndBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _CUOTaskEndAttention(dim)


class _CUOTaskEndViT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([_CUOTaskEndBlock(dim)])
        self.head = nn.Identity()

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block.attn.qkv(inputs)
        return self.head(inputs)


class _CUOCalibrationDataset(Dataset):
    def __init__(self, count=3):
        self.inputs = torch.randn(count, 4, 6)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return index, self.inputs[index], torch.tensor(0)


class _CUOCalibrationDataManager:
    def __init__(self):
        self.calls = []
        self.dataset = _CUOCalibrationDataset()

    def get_task_size(self, task_index):
        assert task_index == 0
        return 2

    def get_dataset(self, classes, source, mode):
        self.calls.append((tuple(classes.tolist()), source, mode))
        return self.dataset


class _CUORawNetwork(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs):
        return self.backbone(inputs)


def _make_tiny_cuo_model(run, task, vit=None):
    return SharedALoRA_ViT_timm(
        copy.deepcopy(vit) if vit is not None else _CUOTaskEndViT(dim=6),
        r=2,
        filepath=str(run),
        cur_task_index=task,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="cuo_lowrank",
        cumulative_rank=2,
        cuo_lambda=0.1,
    )


def _train_tiny_cuo_task_zero(run, vit):
    torch.manual_seed(113)
    model = _make_tiny_cuo_model(run, task=0, vit=vit)
    with torch.no_grad():
        for weight in model.w_As + model.w_Bs:
            weight.weight.copy_(torch.randn_like(weight.weight))
        model.wrapped_param[0].param.fill_(0.63)
    model.prepare_cuo_calibration()
    with torch.no_grad():
        model(torch.randn(5, 4, 6))
    model.finalize_cuo_calibration()
    model.save_lora_parameters(str(run), task_id=0)
    return model


def test_cuo_historical_branch_uses_fixed_projection_after_a_changes():
    wrapper = make_cuo_wrapper(dim=8, rank=3)
    inputs = torch.randn(2, 5, 8)
    before_q, _ = wrapper.historical_output(inputs)
    with torch.no_grad():
        wrapper.a_q.weight.add_(torch.randn_like(wrapper.a_q.weight))
    after_q, _ = wrapper.historical_output(inputs)
    assert torch.allclose(before_q, after_q, atol=1e-6, rtol=1e-6)


def test_cuo_wrapper_collects_fp64_projected_total_residual_statistics():
    torch.manual_seed(19)
    wrapper = make_cuo_wrapper(dim=6, rank=2)
    inputs = torch.randn(3, 4, 6)
    wrapper.begin_cuo_calibration()
    wrapper(inputs)

    (gram_q, cross_q, count_q), (gram_v, cross_v, count_v) = (
        wrapper.consume_cuo_statistics()
    )
    historical_q, historical_v = wrapper.historical_output(inputs)
    current_q, current_v = wrapper.current_output(inputs)
    z_q = inputs.double().reshape(-1, 6) @ wrapper.projection_q.double().T
    z_v = inputs.double().reshape(-1, 6) @ wrapper.projection_v.double().T
    y_q = (historical_q + current_q).double().reshape(-1, 6)
    y_v = (historical_v + current_v).double().reshape(-1, 6)

    assert gram_q.dtype == torch.float64
    assert cross_q.dtype == torch.float64
    assert count_q == 12
    assert torch.allclose(gram_q, z_q.T @ z_q)
    assert torch.allclose(cross_q, y_q.T @ z_q)
    assert gram_v.dtype == torch.float64
    assert cross_v.dtype == torch.float64
    assert count_v == 12
    assert torch.allclose(gram_v, z_v.T @ z_v)
    assert torch.allclose(cross_v, y_v.T @ z_v)


def test_cuo_wrapper_does_not_collect_in_normal_training():
    wrapper = make_cuo_wrapper(dim=6, rank=2).train()
    wrapper(torch.randn(2, 3, 6))
    (gram_q, cross_q, count_q), (gram_v, cross_v, count_v) = (
        wrapper.consume_cuo_statistics()
    )
    assert count_q == 0
    assert count_v == 0
    assert torch.count_nonzero(gram_q) == 0
    assert torch.count_nonzero(cross_q) == 0
    assert torch.count_nonzero(gram_v) == 0
    assert torch.count_nonzero(cross_v) == 0


def test_cuo_state_roundtrip_preserves_deployed_logits(tmp_path):
    run = tmp_path / "cuo-run"
    pristine = _CUOTaskEndViT(dim=6)
    model = _train_tiny_cuo_task_zero(run, pristine)
    inputs = torch.randn(2, 5, 6)
    with torch.no_grad():
        before = model(inputs)

    rebuilt = _make_tiny_cuo_model(run, task=1, vit=pristine)
    with torch.no_grad():
        after = rebuilt(inputs)

    assert torch.allclose(after, before, atol=2e-6, rtol=2e-6)
    state = torch.load(run / SA_STATE_FILENAME, weights_only=True)
    assert set(state) == {
        "version",
        "task_id",
        "rank",
        "merge_mode",
        "projection_down",
        "unified_up",
        "projected_gram",
        "cuo_lambda",
    }
    assert state["version"] == SA_STATE_VERSION_CUO
    assert state["merge_mode"] == "cuo_lowrank"
    assert len(state["projection_down"]) == 2
    assert len(state["unified_up"]) == 2
    assert len(state["projected_gram"]) == 2
    assert not list(run.glob("sa_lora_w_b_*.pt"))

    merged = torch.load(run / SA_MERGED_FILENAME, weights_only=True)
    assert set(merged) == {
        "version",
        "task_id",
        "rank",
        "merge_mode",
        "projection_down",
        "unified_up",
    }


def test_cuo_task_zero_canonicalizes_trained_a_before_calibration(tmp_path):
    model = _make_tiny_cuo_model(tmp_path / "cuo-run", task=0)
    with torch.no_grad():
        for weight in model.w_As:
            weight.weight.copy_(torch.randn_like(weight.weight))
    trained_a = [weight.weight.detach().clone() for weight in model.w_As]

    model.prepare_cuo_calibration()

    wrappers = [block.attn.qkv for block in model.lora_vit.blocks]
    projections = [wrappers[0].projection_q, wrappers[0].projection_v]
    for projection, a in zip(projections, trained_a):
        expected = row_orthonormal_projection(a)
        assert torch.allclose(projection, expected, atol=1e-6, rtol=1e-6)
        assert torch.allclose(
            projection @ projection.T,
            torch.eye(projection.shape[0]),
            atol=1e-6,
            rtol=1e-6,
        )


def test_cuo_finalization_retires_temporary_b_on_every_rank(tmp_path):
    model = _make_tiny_cuo_model(tmp_path / "cuo-run", task=0)
    with torch.no_grad():
        for weight in model.w_Bs:
            weight.weight.copy_(torch.randn_like(weight.weight))
    model.prepare_cuo_calibration()
    with torch.no_grad():
        model(torch.randn(3, 4, 6))

    model.finalize_cuo_calibration()

    assert all(torch.count_nonzero(weight.weight) == 0 for weight in model.w_Bs)


def test_cuo_task_one_loads_p_without_rederiving_it(tmp_path):
    run = tmp_path / "cuo-run"
    pristine = _CUOTaskEndViT(dim=6)
    _train_tiny_cuo_task_zero(run, pristine)
    state = torch.load(run / SA_STATE_FILENAME, weights_only=True)
    model = _make_tiny_cuo_model(run, task=1, vit=pristine)
    wrapper = model.lora_vit.blocks[0].attn.qkv
    before = [wrapper.projection_q.clone(), wrapper.projection_v.clone()]

    model.prepare_cuo_calibration()

    assert torch.allclose(before[0], state["projection_down"][0])
    assert torch.allclose(before[1], state["projection_down"][1])
    assert torch.allclose(wrapper.projection_q, before[0])
    assert torch.allclose(wrapper.projection_v, before[1])
    assert torch.allclose(model.w_As[0].weight, before[0])
    assert torch.allclose(model.w_As[1].weight, before[1])


def test_learner_cuo_calibration_uses_eval_preprocessing_before_save(tmp_path):
    backbone = _make_tiny_cuo_model(tmp_path / "cuo-run", task=0)
    raw_network = _CUORawNetwork(backbone).train()
    data_manager = _CUOCalibrationDataManager()
    learner = object.__new__(SharedALearner)
    learner._cur_task = 0
    learner._known_classes = 0
    learner._total_classes = 2
    learner._device = torch.device("cpu")
    learner.args = {"batch_size": 2}
    learner._cuo_calibration_data_manager = data_manager
    learner._loader_workers = lambda: 0

    SharedALearner._before_task_save(learner, raw_network, train_loader=None)

    assert data_manager.calls == [((0, 1), "train", "test")]
    assert raw_network.training is True
    assert backbone._cuo_calibration_finalized is True
    assert backbone._last_cuo_calibration_stats["token_count"] == 12


def test_cuo_rejects_generic_gauge_state_at_load_boundary(tmp_path):
    run = tmp_path / "legacy-cuo-run"
    run.mkdir()
    projection = torch.cat([torch.eye(2), torch.zeros(2, 4)], dim=1)
    torch.save(
        {
            "version": 2,
            "task_id": 1,
            "rank": 2,
            "canonical_down": [projection, projection.clone()],
            "cumulative_up": [torch.zeros(6, 2), torch.zeros(6, 2)],
            "triangular_r": [torch.eye(2), torch.eye(2)],
        },
        run / "sa_state.pt",
    )

    with pytest.raises(ValueError, match="cuo_lowrank.*version"):
        SharedALoRA_ViT_timm(
            _CUOTaskEndViT(dim=6),
            r=2,
            filepath=str(run),
            cur_task_index=1,
            train_a_all_tasks=True,
            cumulative_state=True,
            cumulative_merge="cuo_lowrank",
            cumulative_rank=2,
            cuo_lambda=0.1,
        )


def test_rank10_vit_state_counts_factors_and_grams():
    counts = cuo_state_scalar_counts(num_blocks=12, rank=10, dim=768)
    assert counts == {
        "lora_factor_scalars": 368640,
        "gram_scalars": 2400,
        "persistent_scalar_total": 371040,
    }


def test_cuo_reducer_combines_rank_local_statistics_once():
    torch.manual_seed(127)
    z0, y0 = torch.randn(7, 3, dtype=torch.float64), torch.randn(
        7, 6, dtype=torch.float64
    )
    z1, y1 = torch.randn(11, 3, dtype=torch.float64), torch.randn(
        11, 6, dtype=torch.float64
    )
    gram, cross, count = combine_cuo_statistics(
        [
            (z0.T @ z0, y0.T @ z0, z0.shape[0]),
            (z1.T @ z1, y1.T @ z1, z1.shape[0]),
        ]
    )
    z, y = torch.cat([z0, z1]), torch.cat([y0, y1])
    direct = torch.linalg.solve(
        z.T @ z + 0.2 * torch.eye(3),
        (y.T @ z).T,
    ).T
    merged, _ = solve_projected_cuo(gram, cross, damping=0.2)

    assert count == z.shape[0]
    assert torch.allclose(gram, z.T @ z, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cross, y.T @ z, atol=1e-6, rtol=1e-6)
    assert torch.allclose(merged, direct, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sa_cumulative_state": False}, "sa_cumulative_state"),
        ({"sa_train_a_all_tasks": False}, "sa_train_a_all_tasks"),
        ({"sa_cumulative_rank": 2}, "sa_cumulative_rank"),
        ({"sa_cuo_lambda": 0.0}, "sa_cuo_lambda"),
        ({"sa_cuo_lambda": float("nan")}, "sa_cuo_lambda"),
        ({"sa_coordinate_stable_transport": True}, "CoordinateStable"),
        ({"sa_live_a_coordinate_align": True}, "CoordinateStable"),
        ({"sa_hbd_enabled": True}, "HBD"),
        ({"sa_adaptive_a_enabled": True}, "Adaptive-A"),
        ({"sa_adaptive_a_strategy": "function_safe_pareto"}, "function-safe"),
    ],
)
def test_cuo_lowrank_rejects_incompatible_configuration(overrides, message):
    config = {
        "sa_cumulative_merge": "cuo_lowrank",
        "sa_cumulative_state": True,
        "sa_train_a_all_tasks": True,
        "lora_rank": 3,
        "sa_cumulative_rank": 3,
        "sa_cuo_lambda": 0.1,
    }
    config.update(overrides)
    with pytest.raises(ValueError, match=message):
        validate_cuo_lowrank_config(config)


def test_cuo_lowrank_accepts_fixed_projection_configuration():
    config = {
        "sa_cumulative_merge": "cuo_lowrank",
        "sa_cumulative_state": True,
        "sa_train_a_all_tasks": True,
        "lora_rank": 3,
        "sa_cumulative_rank": 3,
        "sa_cuo_lambda": 0.1,
    }
    assert validate_cuo_lowrank_config(config) == pytest.approx(0.1)


def test_projected_cuo_matches_direct_concatenated_ridge_solution():
    z0, y0 = torch.randn(13, 3), torch.randn(13, 7)
    z1, y1 = torch.randn(17, 3), torch.randn(17, 7)
    up0, gram0, _ = advance_projected_cuo(
        torch.zeros(7, 3), torch.zeros(3, 3), z0, y0, damping=0.2
    )
    up1, gram1, _ = advance_projected_cuo(up0, gram0, z1, y1, damping=0.2)
    z, y = torch.cat([z0, z1]), torch.cat([y0, y1])
    direct = torch.linalg.solve(
        z.T @ z + 0.2 * torch.eye(3), (y.T @ z).T
    ).T
    assert torch.allclose(up1, direct, atol=1e-6, rtol=1e-6)
    assert torch.allclose(gram1, z.T @ z, atol=1e-6, rtol=1e-6)


def test_projected_cuo_is_additive_over_batches():
    torch.manual_seed(3)
    z0, y0 = torch.randn(8, 4), torch.randn(8, 6)
    z1, y1 = torch.randn(11, 4), torch.randn(11, 6)
    sequential, gram, _ = advance_projected_cuo(
        *advance_projected_cuo(
            torch.zeros(6, 4), torch.zeros(4, 4), z0, y0, damping=0.4
        )[:2],
        z1,
        y1,
        damping=0.4,
    )
    combined, combined_gram, _ = advance_projected_cuo(
        torch.zeros(6, 4), torch.zeros(4, 4), torch.cat([z0, z1]), torch.cat([y0, y1]), damping=0.4
    )
    assert torch.allclose(sequential, combined, atol=1e-6, rtol=1e-6)
    assert torch.allclose(gram, combined_gram, atol=1e-6, rtol=1e-6)


def test_row_orthonormal_projection_uses_reduced_qr():
    projection = row_orthonormal_projection(torch.randn(3, 9))
    assert projection.shape == (3, 9)
    assert torch.allclose(projection @ projection.T, torch.eye(3), atol=1e-6)


@pytest.mark.parametrize("damping", [0.0, -0.1, float("nan"), float("inf")])
def test_projected_cuo_requires_positive_finite_damping(damping):
    with pytest.raises(ValueError, match="damping"):
        solve_projected_cuo(torch.eye(2), torch.ones(3, 2), damping)


def test_projected_cuo_rejects_invalid_shapes_and_nonsymmetric_gram():
    with pytest.raises(ValueError, match="shape"):
        solve_projected_cuo(torch.eye(2), torch.ones(3, 3), damping=0.1)
    with pytest.raises(ValueError, match="symmetric"):
        solve_projected_cuo(torch.tensor([[1.0, 2.0], [0.0, 1.0]]), torch.ones(3, 2), damping=0.1)


def test_projected_cuo_preserves_cross_dtype_and_device():
    gram = torch.eye(2, dtype=torch.float32)
    cross = torch.randn(5, 2, dtype=torch.float32)
    up, diagnostics = solve_projected_cuo(gram, cross, damping=0.1)
    assert up.dtype == cross.dtype
    assert up.device == cross.device
    assert diagnostics["condition_number"] > 0


def test_projected_cuo_rejects_nonfinite_statistics():
    with pytest.raises(ValueError, match="finite"):
        solve_projected_cuo(torch.tensor([[float("nan")]]), torch.ones(2, 1), damping=0.1)


def test_projected_cuo_handles_ill_conditioned_gram_with_damping():
    gram = torch.diag(torch.tensor([1e-12, 1.0]))
    cross = torch.ones(3, 2)
    up, _ = solve_projected_cuo(gram, cross, damping=0.2)
    assert torch.isfinite(up).all()
