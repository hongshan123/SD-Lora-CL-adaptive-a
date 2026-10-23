"""Read-only boundary captures. No old examples are used by training."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from utils.rng_utils import rng_preserving, rng_state_hash


def cpu(x):
    return x.detach().cpu().clone()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe(backbone, inputs):
    modes=[(m,m.training) for m in backbone.modules()]
    try:
        backbone.eval()
        with rng_preserving(),torch.no_grad():
            return cpu(backbone(inputs.to(next(backbone.parameters()).device)))
    finally:
        for m,mode in modes:m.training=mode


def capture(learner, raw):
    cfg=learner.args
    if cfg.get('distributed') and int(cfg.get('rank',0)) != 0:
        return
    before=rng_state_hash()
    if cfg.get('sa_live_a_absorb_mode')!='operator_preserving_absorb':
        raise ValueError('HOEP diagnosis requires exact absorption')
    b=raw.backbone; task=learner._cur_task
    root=Path(cfg['filepath']); out=root/'task_snapshots'/f'task_{task:03d}'
    out.mkdir(parents=True,exist_ok=True)
    old=torch.load(root/'sa_prototypes.pt',map_location='cpu',weights_only=True) if task else {}
    branches=[]
    for block in b.lora_vit.blocks:
        w=block.attn.qkv
        for suffix in ['q','v']:
            branches.append({'down':cpu(getattr(w,'a_'+suffix).weight),
                             'current_up':cpu(getattr(w,'b_'+suffix).weight),
                             'historical_up':cpu(getattr(w,'aggregate_'+suffix))})
    manager=learner._cuo_calibration_data_manager
    with rng_preserving():
        dataset=manager.get_dataset(np.arange(learner._known_classes,learner._total_classes),source='train',mode='test')
        samples=[dataset[i] for i in range(min(8,len(dataset)))]
        inputs=torch.stack([x[1] for x in samples])
        features=probe(b,inputs)
    snap={'version':4,'stage':'pre_merge','task_id':task,
          'known_classes':learner._known_classes,'total_classes':learner._total_classes,
          'branches':branches,'current_scale':cpu(b.wrapped_param[0].param).reshape(()),
          'normalize_current_branch':False,'history_forward':'shared',
          'absorb_mode':cfg['sa_live_a_absorb_mode'],'old_prototypes':old,
          'probe_inputs':inputs,'probe_features':features,
          'probe_source':'first 8 current-task training images; test preprocessing',
          'fc_weight':cpu(raw.fc.weight),'fc_bias':cpu(raw.fc.bias)}
    if task:
        snap['alignment_previous']={'shared_a':[cpu(x) for x in b._live_a_previous_shared_a],
                                    'aggregate_up':[cpu(x) for x in b.aggregate_up]}
    torch.save(snap,out/'pre_merge.pt')
    (out/'config.json').write_text(json.dumps(cfg,default=str,indent=2)+'\n')
    if before!=rng_state_hash():raise RuntimeError('snapshot changed RNG')


def audit_snapshot(directory):
    d=Path(directory);record=json.loads((d/'complete.json').read_text())
    for name,expected in record['sha256'].items():
        if sha(d/name)!=expected:raise RuntimeError(f'checksum mismatch: {d/name}')
    return record


def finalize(learner, raw):
    if learner.args.get('distributed') and int(learner.args.get('rank',0)) != 0:
        return
    before=rng_state_hash(); root=Path(learner.args['filepath']);task=learner._cur_task
    out=root/'task_snapshots'/f'task_{task:03d}'
    for name in ['sa_state.pt','sa_merged_lora.pt','sa_prototypes.pt']:
        shutil.copy2(root/name,out/name)
    for name in [f'CLs_weight{task}.pt',f'CLs_bias{task}.pt','sa_coordinate_diagnostics.json','hoep_partition_schedule.json','run_manifest.json']:
        if (root/name).is_file():shutil.copy2(root/name,out/name)
    pre=torch.load(out/'pre_merge.pt',map_location='cpu',weights_only=True)
    torch.save(probe(raw.backbone,pre['probe_inputs']),out/'deployed_probe_features.pt')
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}
    marker = out/'complete.json.tmp'
    marker.write_text(json.dumps({'task_id':task,'sha256':hashes},indent=2)+'\n')
    os.replace(marker, out/'complete.json')
    if before!=rng_state_hash():raise RuntimeError('post snapshot changed RNG')
    if task and learner.args.get('sa_boundary_diagnose',True):
        # Separate process isolates RNG, data and model construction from training.
        torch.cuda.empty_cache()
        destination=root/'diagnosis';destination.mkdir(exist_ok=True)
        command=[sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/diagnose_hoep.py'),
                 '--run-dir',str(root),'--task',str(task),'--output-dir',str(destination),
                 '--batch-size',str(learner.args.get('sa_diagnostic_batch_size',64))]
        with (destination/f'task_{task:03d}.log').open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if before!=rng_state_hash():raise RuntimeError('offline diagnosis changed training RNG')
