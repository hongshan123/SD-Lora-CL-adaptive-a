"""Checks that direction controls preserve dimensions, ties and training RNG."""

import json
from pathlib import Path
import copy
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backbone import operator_energy_partition as energy
from models.sa_sdlora import validate_adaptive_a_config
from backbone.sa_lora import SharedALoRA_ViT_timm
from test_operator_energy_partition import _TinyViT


def select(*args, **kwargs):
    implementation = getattr(energy, "select_matched_energy_partition", None)
    assert callable(implementation), "matched-count direction controls are missing"
    return implementation(*args, **kwargs)


def test_high_energy_control_preserves_counts_and_selects_largest_energy():
    spectra = [torch.tensor([1., 2., 3., 4.]), torch.tensor([1., 2., 3.])]
    result = select(spectra, [2, 1], "high", seed=1)
    assert result.plastic_masks[0].tolist() == [False, False, True, True]
    assert result.plastic_masks[1].tolist() == [False, False, True]
    assert result.selected_energy == pytest.approx(10.)
    assert result.selected_directions == 3


def test_random_control_is_reproducible_and_does_not_consume_training_rng():
    spectra = [torch.arange(1., 11.) for _ in range(8)]
    before = torch.get_rng_state().clone()
    first = select(spectra, [3] * 8, "random", seed=1993)
    assert torch.equal(before, torch.get_rng_state())
    second = select(spectra, [3] * 8, "random", seed=1993)
    third = select(spectra, [3] * 8, "random", seed=1994)
    assert all(int(mask.sum()) == 3 for mask in first.plastic_masks)
    assert all(torch.equal(a, b) for a, b in zip(first.plastic_masks, second.plastic_masks))
    assert any(not torch.equal(a, b) for a, b in zip(first.plastic_masks, third.plastic_masks))


@pytest.mark.parametrize("policy", ["random", "high"])
def test_controls_keep_complete_ties_and_reject_impossible_counts(policy):
    spectra = [torch.tensor([1., 1., 3., 3.])]
    result = select(spectra, [2], policy, seed=2)
    mask = result.plastic_masks[0]
    assert bool(mask[0]) == bool(mask[1])
    assert bool(mask[2]) == bool(mask[3])
    with pytest.raises(ValueError, match="tie"):
        select(spectra, [1], policy, seed=2)


def test_all_plastic_geometry_control_opens_every_direction():
    result = select([torch.tensor([0., 1., 4.])], None, "all", seed=0)
    assert result.plastic_masks[0].tolist() == [True, True, True]
    assert result.selected_energy == pytest.approx(5.)


def test_validation_rejects_unmatched_random_policy():
    config = {
        "sa_adaptive_a_enabled": True,
        "sa_adaptive_a_strategy": "operator_energy_partition",
        "sa_train_a_all_tasks": True,
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": True,
        "sa_live_a_absorb_mode": "operator_preserving_absorb",
        "sa_hoep_partition_policy": "random",
    }
    with pytest.raises(ValueError, match="reference_counts"):
        validate_adaptive_a_config(config)


@pytest.mark.parametrize("policy", ["random", "high", "all"])
def test_control_survives_training_boundary_and_preserves_stable_rows(tmp_path, policy):
    common = dict(train_a_all_tasks=True, cumulative_state=True,
                  cumulative_merge="live_a_aggregate_b", live_a_coordinate_align=True,
                  live_a_absorb_mode="operator_preserving_absorb", adaptive_a_enabled=True,
                  adaptive_a_strategy="operator_energy_partition")
    torch.manual_seed(80)
    previous = SharedALoRA_ViT_timm(_TinyViT(), r=2, filepath=str(tmp_path),
                                 cur_task_index=0, **common)
    with torch.no_grad():
        for b in previous.w_Bs:
            b.weight.normal_()
    previous.save_lora_parameters(str(tmp_path), 0)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"version": 1, "seed": 1993, "tasks": {"1": [1, 1]}}))
    model = SharedALoRA_ViT_timm(_TinyViT(), r=2, filepath=str(tmp_path),
                               cur_task_index=1, index=False, hoep_partition_policy=policy,
                               hoep_reference_counts_path=str(reference),
                               hoep_partition_seed=1993, **common)
    wanted_counts = [2, 2] if policy == "all" else [1, 1]
    assert [int(m.sum()) for m in model._hoep_plastic_masks] == wanted_counts
    anchors = [a.weight.detach().clone() for a in model.w_As]
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.01)
    model(torch.randn(3, 2, 4)).square().mean().backward()
    model.apply_operator_energy_partition_gradients(optimizer)
    optimizer.step()
    model.apply_operator_energy_partition_step(optimizer)
    for module, anchor, mask in zip(model.w_As, anchors, model._hoep_plastic_masks):
        assert torch.equal(module.weight[~mask], anchor[~mask])
        assert torch.allclose(module.weight @ module.weight.T, torch.eye(2), atol=2e-6)
    model.save_lora_parameters(str(tmp_path), 1)
    diag = model._last_live_a_coordinate_diagnostics
    assert diag["global_squared_residual_ratio"] <= diag["predicted_energy_budget"] + 1e-6


@pytest.mark.parametrize("policy", ["low", "random", "high", "all"])
def test_eval_rebuild_uses_saved_operator_without_next_task_partition(tmp_path, policy):
    common = dict(train_a_all_tasks=True, cumulative_state=True,
                  cumulative_merge="live_a_aggregate_b", live_a_coordinate_align=True,
                  live_a_absorb_mode="operator_preserving_absorb", adaptive_a_enabled=True,
                  adaptive_a_strategy="operator_energy_partition")
    previous = SharedALoRA_ViT_timm(_TinyViT(), r=2, filepath=str(tmp_path),
                                 cur_task_index=0, **common)
    with torch.no_grad():
        for b in previous.w_Bs:
            b.weight.normal_()
    previous.save_lora_parameters(str(tmp_path), 0)
    # Task10 means ten absorbed tasks when constructing the final evaluation
    # model. There is no Task10 training partition or matching count schedule.
    state_path = tmp_path / 'sa_state.pt'
    state = torch.load(state_path, weights_only=True)
    state['task_id'] = 10
    torch.save(state, state_path)
    base = _TinyViT()
    original_qkv = copy.deepcopy(base.blocks[0].attn.qkv)
    rebuilt = SharedALoRA_ViT_timm(
        base, r=2, filepath=str(tmp_path), cur_task_index=10, index=False,
        initialize_training_partition=False,
        hoep_partition_policy=policy,
        hoep_reference_counts_path=str(tmp_path / 'no_task10_reference.json'),
        hoep_partition_seed=1993, **common)
    for actual, saved in zip(rebuilt.w_As, state['shared_a']):
        assert torch.equal(actual.weight, saved)
    assert all(torch.equal(a, b) for a, b in zip(rebuilt.aggregate_up, state['aggregate_up']))
    inputs = torch.randn(3, 2, 4)
    expected = original_qkv(inputs)
    for branch, sl in [(0, slice(0, 4)), (1, slice(-4, None))]:
        a, g = state['shared_a'][branch], state['aggregate_up'][branch]
        expected[:, :, sl] += torch.nn.functional.linear(
            torch.nn.functional.linear(inputs, a), g / (a.norm() + 1e-8))
    torch.testing.assert_close(rebuilt(inputs), expected, atol=2e-6, rtol=2e-6)
    assert not rebuilt._hoep_plastic_masks

@pytest.mark.parametrize('deployment', [False, True])
def test_learner_training_and_deployment_rebuild_use_distinct_partitions(tmp_path, monkeypatch, deployment):
    from torch import nn
    from models.sa_sdlora import Learner
    import models.sa_sdlora as learner_module
    common = dict(train_a_all_tasks=True, cumulative_state=True,
                  cumulative_merge='live_a_aggregate_b', live_a_coordinate_align=True,
                  live_a_absorb_mode='operator_preserving_absorb', adaptive_a_enabled=True,
                  adaptive_a_strategy='operator_energy_partition')
    initial = SharedALoRA_ViT_timm(_TinyViT(), r=2, filepath=str(tmp_path), cur_task_index=0, **common)
    with torch.no_grad():
        for b in initial.w_Bs:
            b.weight.normal_()
    initial.save_lora_parameters(str(tmp_path), 0)
    state = torch.load(tmp_path / 'sa_state.pt', weights_only=True)
    reference = tmp_path / 'reference.json'
    reference.write_text(json.dumps({'version':1, 'seed':1993, 'tasks':{'1':[1,1]}}))
    learner = Learner.__new__(Learner)
    learner.args = {'increment':10, 'filepath':str(tmp_path), 'lora_rank':2,
                    **{'sa_'+key:value for key,value in common.items()},
                    'sa_hoep_partition_policy':'high',
                    'sa_hoep_reference_counts_path':str(reference), 'sa_hoep_partition_seed':1993}
    learner._device = torch.device('cpu')
    learner._network = nn.Module()
    learner._network.backbone = initial
    # Replace only the external pretrained model download with a real tiny
    # module. Both production construction paths and partition logic run.
    monkeypatch.setattr(learner_module.timm, 'create_model', lambda *a, **kw: _TinyViT())
    if deployment:
        learner._cur_task = 0
        learner._rebuild_eval_backbone()
        model = learner._network.backbone
        assert not model._hoep_plastic_masks
        assert all(torch.equal(a.weight, b) for a,b in zip(model.w_As,state['shared_a']))
        assert all(torch.equal(a,b) for a,b in zip(model.aggregate_up,state['aggregate_up']))
    else:
        learner._cur_task = 1
        model = learner.update_network(index=False)
        assert [int(mask.sum()) for mask in model._hoep_plastic_masks] == [1,1]
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        model(torch.randn(2,3,4)).square().mean().backward()
        model.apply_operator_energy_partition_gradients(optimizer)
        optimizer.step()
        model.apply_operator_energy_partition_step(optimizer)
