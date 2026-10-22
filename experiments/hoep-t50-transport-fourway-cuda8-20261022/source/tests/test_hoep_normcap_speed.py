import copy
import numpy as np
import pytest
import torch
from torchvision import transforms

from models.sa_sdlora import validate_adaptive_a_config
from backbone.sa_lora import SharedALoRA_ViT_timm, absorb_live_a_current_projection
from backbone.coordinate_stability import align_live_a_aggregate
from test_operator_energy_partition import _TinyViT, _valid_config
from utils.data_manager import DataManager, DummyDataset
from utils.rng_utils import rng_state_hash


def test_normcap_is_allowed_for_hoep_but_normalized_branch_stays_rejected():
    validate_adaptive_a_config(_valid_config(sa_live_a_absorb_mode='bounded_norm_calibrated_absorb', sa_coordinate_stable_transport=True))
    with pytest.raises(ValueError):
        validate_adaptive_a_config(_valid_config(sa_live_a_absorb_mode='normalized_absorb'))
    with pytest.raises(ValueError):
        validate_adaptive_a_config(_valid_config(sa_normalize_current_branch=True))


@pytest.mark.parametrize('interval', [0, -1, True, 1.5])
def test_diagnostic_interval_requires_positive_integer(interval):
    with pytest.raises(ValueError, match='diagnostics_interval'):
        validate_adaptive_a_config(_valid_config(sa_hoep_diagnostics_interval=interval))


def common():
    return dict(train_a_all_tasks=True, cumulative_state=True, cumulative_merge='live_a_aggregate_b',
                live_a_coordinate_align=True, live_a_absorb_mode='bounded_norm_calibrated_absorb',
                adaptive_a_enabled=True, adaptive_a_strategy='operator_energy_partition', hoep_energy_budget=0.5)


def task_one(path, interval=1):
    torch.manual_seed(90)
    previous=SharedALoRA_ViT_timm(_TinyViT(), r=2, filepath=str(path), cur_task_index=0, **common())
    with torch.no_grad():
        for b in previous.w_Bs: b.weight.normal_()
    previous.save_lora_parameters(str(path),0)
    return SharedALoRA_ViT_timm(_TinyViT(),r=2,filepath=str(path),cur_task_index=1,index=False,
                               hoep_diagnostics_interval=interval,**common())


def test_sampled_diagnostics_preserve_every_weight_momentum_rng_and_logit(tmp_path):
    full=task_one(tmp_path/'full'); fast=copy.deepcopy(full)
    fast.hoep_diagnostics_interval=16
    opt1=torch.optim.SGD(full.parameters(),lr=.01,momentum=.9,weight_decay=.01)
    opt2=torch.optim.SGD(fast.parameters(),lr=.01,momentum=.9,weight_decay=.01)
    for step in range(4):
        x=torch.randn(3,2,4)
        for model,opt in ((full,opt1),(fast,opt2)):
            opt.zero_grad(); model(x).square().mean().backward()
            model.apply_operator_energy_partition_gradients(opt);opt.step()
            before=rng_state_hash(); model.apply_operator_energy_partition_step(opt)
            assert before==rng_state_hash()
        for p,q in zip(full.parameters(),fast.parameters()):
            assert torch.equal(p,q)
            if 'momentum_buffer' in opt1.state[p]:
                assert torch.equal(opt1.state[p]['momentum_buffer'],opt2.state[q]['momentum_buffer'])
        assert torch.equal(full(x),fast(x))
    assert full.adaptive_a_diagnostics()['operator_diagnostic_steps']==4
    assert fast.adaptive_a_diagnostics()['operator_diagnostic_steps']==1


def test_normcap_write_matches_formula_and_rebuilt_deployment(tmp_path):
    model=task_one(tmp_path)
    with torch.no_grad():
        for b in model.w_Bs: b.weight.normal_(std=3.)
    expected=[]
    for i,(a,b) in enumerate(zip(model.w_As,model.w_Bs)):
        history,_=align_live_a_aggregate(model.aggregate_up[i],model._live_a_previous_shared_a[i],a.weight.detach())
        current,diag=absorb_live_a_current_projection(a.weight.detach(),b.weight.detach(),model.wrapped_param[0].param.detach(),mode='bounded_norm_calibrated_absorb')
        assert diag['consolidation_gain']<1.
        expected.append(history+current)
    model.save_lora_parameters(str(tmp_path),1)
    state=torch.load(tmp_path/'sa_state.pt',weights_only=True)
    for actual,wanted in zip(state['aggregate_up'],expected):
        torch.testing.assert_close(actual,wanted,atol=1e-6,rtol=1e-6)
    assert model._last_live_a_coordinate_diagnostics['global_squared_residual_ratio']<=0.5+1e-6
    torch.manual_seed(123)
    base=_TinyViT(); oracle=copy.deepcopy(base)
    rebuilt=SharedALoRA_ViT_timm(base,r=2,filepath=str(tmp_path),cur_task_index=2,index=False,
                                initialize_training_partition=False,**common())
    x=torch.randn(2,3,4);wanted=oracle(x)
    for i,sl in ((0,slice(0,4)),(1,slice(-4,None))):
        a=state['shared_a'][i];g=state['aggregate_up'][i]
        wanted[:,:,sl]+=torch.nn.functional.linear(torch.nn.functional.linear(x,a),g/(a.norm()+1e-8))
    torch.testing.assert_close(rebuilt(x),wanted,atol=2e-6,rtol=2e-6)


def data_manager():
    dm=DataManager.__new__(DataManager)
    dm.args={'sa_cache_eval_inputs':True,'sa_eval_cache_gib':0.001}
    dm.dataset_name='cifar224';dm.use_path=False
    dm._train_data=dm._test_data=np.arange(4*4*4*3,dtype=np.uint8).reshape(4,4,4,3)
    dm._train_targets=dm._test_targets=np.array([0,0,1,1])
    dm._train_trsf=[transforms.RandomHorizontalFlip()]
    dm._test_trsf=[transforms.ToTensor()];dm._common_trsf=[]
    return dm


def test_eval_cache_reuses_overlapping_subsets_without_changing_tensors_or_rng():
    dm=data_manager();first=dm.get_dataset([0],source='test',mode='test')
    before=rng_state_hash(); original=first[0][1].clone();first[0][1].zero_()
    bigger=dm.get_dataset([0,1],source='test',mode='test')
    assert torch.equal(bigger[0][1],original)
    assert rng_state_hash()==before
    assert dm._eval_input_cache.hits==2
    assert dm.get_dataset([0],source='train',mode='train').input_cache is None


def test_eval_cache_is_bounded_and_lru_eviction_keeps_values_exact():
    from utils.eval_input_cache import EvalInputCache
    cache=EvalInputCache(16)
    cache.put('a',torch.arange(4.));cache.put('b',torch.ones(4))
    assert cache.get('a') is None and cache.bytes_used==16
    assert torch.equal(cache.get('b'),torch.ones(4))
    cache.put('oversized',torch.ones(5));assert cache.bytes_used==16
