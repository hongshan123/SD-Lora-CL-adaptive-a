import json
import sys
from pathlib import Path
import torch
old,new=map(Path,sys.argv[1:])
a=torch.load(old/'task_001.pt',map_location='cpu',weights_only=True)
b=torch.load(new/'task_001.pt',map_location='cpu',weights_only=True)
assert torch.equal(a['sample_ids'],b['sample_ids']) and torch.equal(a['labels'],b['labels'])
assert set(a['logits'])==set(b['logits'])
errors={}
for key in a['logits']:
    x,y=a['logits'][key],b['logits'][key]
    err=float((x-y).abs().max());rel=float((x-y).norm()/x.norm().clamp_min(1e-12))
    assert err<1e-4 and rel<1e-5,(key,err,rel)
    assert torch.equal(x.argmax(1),y.argmax(1)),key
    errors[key]={'max_abs':err,'relative_l2':rel}
a=json.loads((old/'task_001.json').read_text());b=json.loads((new/'task_001.json').read_text())
for branch_a,branch_b in zip(a['branches'],b['branches']):
    x,y=branch_a['r_weighted'],branch_b['r_weighted']
    assert abs(x-y)<=1e-5*max(abs(x),1e-12)+1e-12,(x,y)
assert a['sample_count']==b['sample_count']
print(json.dumps({'passed':True,'logits_differences':errors,'weighted_old':a['aggregate']['r_weighted'],'weighted_new':b['aggregate']['r_weighted']},indent=2))
