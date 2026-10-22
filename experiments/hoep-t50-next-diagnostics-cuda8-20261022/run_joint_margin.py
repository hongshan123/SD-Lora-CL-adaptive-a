"""Oracle-informed joint class-margin transport gate on the fixed HOEP cache.

Optimizes only a predeclared training-centroid risk; never reads test logits to
choose the gates. Requires refreshed old-training-data centroids at each task.
"""
import json
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F

import diagnose_next as dn

ROOT=Path(__file__).resolve().parent
NAME='oracle_joint_margin'


def choices_joint(old,mapping,q):
    old=F.normalize(old.float(),dim=1)
    q=F.normalize(q.float(),dim=1)
    n=len(old)
    alphas=torch.tensor(dn.META['oracle_alphas'])
    moved=dn.fw.moved(old,mapping)
    candidate=F.normalize(old[:,None,:]+alphas[None,:,None]*(moved-old)[:,None,:],dim=2)
    # projection[source_class, updated_class, alpha]
    error=candidate-q[:n,None,:]
    projections=(q[:n]@error.reshape(n*len(alphas),-1).T).reshape(n,n,len(alphas))
    similarity=q@q.T
    similarity.fill_diagonal_(-float('inf'))
    neighbors=similarity.topk(min(5,len(q)-1),dim=1).indices[:n]
    source=torch.arange(n).repeat_interleave(neighbors.shape[1])
    target=neighbors.reshape(-1)
    margin=(q[source]*(q[source]-q[target])).sum(1).clamp_min(1e-6)
    # Cosine initialization improves stability of coordinate descent. It is
    # a training-only oracle criterion and uses the same five candidates.
    index=(1-(candidate*q[:n,None,:]).sum(2)).argmin(1)

    def objective(selected):
        own=projections[source,source,selected[source]]
        neighbor=torch.zeros_like(own)
        mask=target<n
        neighbor[mask]=projections[source[mask],target[mask],selected[target[mask]]]
        return float(((neighbor-own).clamp_min(0)/margin).square().mean())

    initial=objective(index)
    for _ in range(3):
        changes=0
        for c in range(n):
            affected=(source==c)|(target==c)
            s=source[affected]
            j=target[affected]
            self_value=projections[s,s,index[s]][:,None].expand(-1,len(alphas)).clone()
            mask_s=s==c
            self_value[mask_s]=projections[c,c][None]
            other=torch.zeros_like(self_value)
            old_neighbor=j<n
            if old_neighbor.any():
                other[old_neighbor]=projections[s[old_neighbor],j[old_neighbor],
                                                 index[j[old_neighbor]]][:,None]
            mask_j=j==c
            other[mask_j]=projections[s[mask_j],c,:]
            loss=(((other-self_value).clamp_min(0)/margin[affected,None]).square()).sum(0)
            best=int(loss.argmin())
            changes+=best!=int(index[c])
            index[c]=best
        if changes==0:break
    final=objective(index)
    assert final<=initial+1e-7,(initial,final)
    chosen=candidate[torch.arange(n),index]
    return chosen,alphas[index],dict(initial_joint_centroid_risk=initial,
                                    final_joint_centroid_risk=final)


def main():
    dn.fw.setup()
    dn.write(ROOT/'joint_status.json',dict(state='running',started=time.time()))
    kind=NAME
    prototypes=torch.empty((0,768))
    curve=[];matrix=[];records=[]
    try:
        for task in range(50):
            data=dn.load_data(task)
            previous=dn.fw.load(dn.PREV/'experiments/none'/f'task_{task:03d}.pt')
            if task:
                old_q=dn.reference_old(task)
                q=F.normalize(torch.stack([old_q[c] for c in range(2*task)]+
                      [data['new_prototypes'][c] for c in range(2*task,2*task+2)]),dim=1)
                old,alpha,extra=choices_joint(prototypes,data['maps']['primary'],q)
            else:
                old,alpha,extra=prototypes,torch.empty(0),{}
            prototypes,scores,record=dn.metric_step(kind,task,data,old,alpha,previous,extra)
            curve.append(record['global_top1']);records.append(record)
            labels=data['test_labels']
            matrix.append([100*float((scores.argmax(1)[labels//2==j]==labels[labels//2==j]).double().mean())
                           for j in range(task+1)])
            dn.summarize(kind,curve,matrix,records)
            print('JOINT_MARGIN',task,curve[-1],extra,flush=True)
        dn.write(ROOT/'joint_status.json',dict(state='complete',finished=time.time()))
    except BaseException:
        dn.write(ROOT/'joint_status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()))
        raise


if __name__=='__main__':main()
