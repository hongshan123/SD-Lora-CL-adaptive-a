import copy
import torch
from utils.diagnostic_microbatch import microbatch_backward
from scripts.hoep_stages import build_stage_factors, effective_operator

def test_unequal_microbatch_sgd_equivalence():
    torch.manual_seed(7)
    a=torch.nn.Linear(5,3).double(); b=copy.deepcopy(a)
    x=torch.randn(11,5,dtype=torch.double);y=torch.arange(11)%3
    oa=torch.optim.SGD(a.parameters(),lr=.02,momentum=.9)
    ob=torch.optim.SGD(b.parameters(),lr=.02,momentum=.9)
    for _ in range(2):
        oa.zero_grad();torch.nn.functional.cross_entropy(a(x),y).backward();oa.step()
        ob.zero_grad()
        scores,loss,extras=microbatch_backward(lambda xx,yy:(b(xx),torch.nn.functional.cross_entropy(b(xx),yy),{}),x,y,4)
        ob.step()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=1e-12,atol=1e-12)

def test_gauge_changed_history_and_exact_deployment():
    torch.manual_seed(4)
    old=torch.linalg.qr(torch.randn(6,2)).Q.T
    g=torch.randn(4,2);rot=torch.linalg.qr(torch.randn(2,2)).Q
    start=rot.T@old;canonical=g@rot
    current=torch.linalg.qr((start+.03*torch.randn_like(start)).T).Q.T
    actual=canonical+.01*torch.randn_like(canonical);b=torch.randn(4,2)*.02
    previous={'shared_a':[old],'aggregate_up':[g]}
    pre={'task_id':1,'current_scale':torch.tensor(.8),'branches':[{'down':current,'current_up':b,'historical_up':actual}],
         'alignment_previous':{'shared_a':[start],'aggregate_up':[canonical]},'absorb_mode':'operator_preserving_absorb'}
    states,diags=build_stage_factors(previous,pre)
    torch.testing.assert_close(effective_operator(states['S0'],0),g@old/(old.norm()+1e-8))
    torch.testing.assert_close(effective_operator(states['S1'],0),actual@current/(current.norm()+1e-8))
    torch.testing.assert_close(effective_operator(states['S2'],0),actual@current/(current.norm()+1e-8)+.8*b@current)
    torch.testing.assert_close(effective_operator(states['S3'],0),effective_operator(states['S4'],0))
    torch.testing.assert_close(effective_operator(states['S4'],0),effective_operator(states['S5'],0))


def test_cached_features_same_logits_and_predictions():
    from scripts.diagnose_hoep import scores_from_cached_features
    torch.manual_seed(9)
    features=torch.randn(139,32)
    head=torch.nn.functional.normalize(torch.randn(17,32),dim=1)
    actual=scores_from_cached_features(features,head,torch.device('cpu'),64)
    reference=torch.cat([torch.nn.functional.normalize(f,dim=1)@head.T for f in features.split(64)])
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)


def test_weighted_lowrank_matches_dense_residual():
    torch.manual_seed(5)
    a=torch.linalg.qr(torch.randn(24,4,dtype=torch.float64)).Q.T
    new=torch.linalg.qr((a+.002*torch.randn_like(a)).T).Q.T
    g=torch.randn(24,4,dtype=torch.float64)
    x=torch.randn(100,24,dtype=torch.float32)
    down=a/(a.norm()+1e-8)
    projection=torch.linalg.pinv(new)@new
    lost=down@(torch.eye(24,dtype=torch.float64)-projection)
    exact=(x.double()@(g@lost).T).square().sum()
    fast=((x@lost.float().T)@g.float().T).square().sum(dtype=torch.float64)
    torch.testing.assert_close(fast,exact,rtol=1e-5,atol=1e-10)
