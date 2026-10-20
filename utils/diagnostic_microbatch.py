"""One optimizer update per logical batch; weight short microbatches by sample count."""
import torch

def microbatch_backward(forward_loss, inputs, targets, size):
    if size <= 0 or len(inputs) != len(targets) or len(targets) == 0:
        raise ValueError("invalid microbatch inputs")
    scores=[]; total_loss=0.; extras={}
    for start in range(0,len(targets),size):
        xx,yy=inputs[start:start+size],targets[start:start+size]
        logits,loss,additional=forward_loss(xx,yy)
        weight=len(yy)/len(targets)
        (loss*weight).backward()
        scores.append(logits.detach())
        total_loss+=float(loss.detach())*weight
        for name,value in additional.items():extras[name]=extras.get(name,0.)+float(value.detach())*weight
    return torch.cat(scores),total_loss,extras
