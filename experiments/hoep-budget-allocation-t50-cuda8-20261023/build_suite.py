"""Prepare and freeze the new allocation suite; does not start any models."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
MAIN=ROOT.parent/'hoep-t50-improvement-cuda8-20261023'
REFERENCE=ROOT.parent/'hoep-a-step-t50-cuda8-20261023'
CHANGED={'backbone/sa_lora.py','models/sa_sdlora.py','utils/inc_net.py'}
EXTRA={'backbone/budget_allocation.py'}


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def choose(manifest,dataset,arm,phase):
    return next(j for j in manifest['jobs'] if j['dataset']==dataset
                and j['arm']==arm and j['phase']==phase)


def main():
    if (ROOT/'manifest.json').exists() or (ROOT/'status.json').exists():
        raise FileExistsError('Frozen queue inputs already exist')
    for folder in ['configs','outputs','logs','proofs','records']:(ROOT/folder).mkdir(exist_ok=True)
    reference=read(REFERENCE/'manifest.json');parent=read(MAIN/'manifest.json')
    for relative,digest in reference['file_sha256'].items():
        assert sha(REFERENCE/relative)==digest,relative
    reference_files={p.relative_to(REFERENCE/'source') for p in (REFERENCE/'source').rglob('*.py')}
    actual_files={p.relative_to(ROOT/'source') for p in (ROOT/'source').rglob('*.py')}
    assert {str(p) for p in actual_files-reference_files}==EXTRA
    assert not reference_files-actual_files
    changed=set();patch=[]
    for relative in sorted(reference_files):
        before=(REFERENCE/'source'/relative).read_bytes();after=(ROOT/'source'/relative).read_bytes()
        if before!=after:
            changed.add(str(relative))
            patch.extend(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),
                         fromfile='reference/'+str(relative),tofile='allocation/'+str(relative)))
    assert changed==CHANGED,changed
    (ROOT/'proofs/training_changes.diff').write_text(''.join(patch))
    write(ROOT/'proofs/source_scope.json',dict(reference=str(REFERENCE),changed=sorted(changed),
          added=sorted(EXTRA),unchanged_files=len(reference_files)-len(changed)))
    for p in (ROOT/'source').rglob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                         cwd=ROOT,capture_output=True,text=True)
    output=tests.stdout+tests.stderr;(ROOT/'proofs/unit_tests.log').write_text(output)
    assert tests.returncode==0,output
    assert re.search(r'Ran 14 tests',output),output
    write(ROOT/'proofs/unit_tests_passed.json',dict(passed=True,tests=14,
          python=sys.executable,log_sha256=sha(ROOT/'proofs/unit_tests.log')))
    data=(REFERENCE/'source/data').resolve()
    for ds in ('cub','imagenet-r'):
        for split in ('train','test'):assert (data/ds/split).is_dir()
    link=ROOT/'source/data'
    if link.exists() or link.is_symlink():assert link.resolve()==data
    else:link.symlink_to(data,target_is_directory=True)
    sys.path.insert(0,str(ROOT/'source'));sys.path.insert(0,str(ROOT))
    import torch
    from backbone.budget_allocation import select_with_branch_coverage
    from allocation_checks import validate_allocation_diagnostic
    torch.set_num_threads(1)
    probe=read(MAIN/'records/allocation_capacity_probe.json')
    expected=read(MAIN/'records/allocation_coverage_probe.json')
    probe_checks=[]
    for item,wanted in zip(probe['rows'],expected['rows']):
        path=Path(next(p for p in item['provenance'] if p.endswith('/pre_merge.pt'))).parent
        original=read(path/'hoep_partition_schedule.json')['diagnostics'][str(item['task_id'])]
        spectra=[torch.tensor(v,dtype=torch.float64) for v in original['branch_spectra']]
        p=select_with_branch_coverage(spectra,.05,1e-6,1e-8)
        assert [m.tolist() for m in p.plastic_masks]==wanted['branch_masks'],item['job']
        d=copy.deepcopy(original)
        d.update(branch_masks=[m.tolist() for m in p.plastic_masks],
                 branch_counts=[int(m.sum()) for m in p.plastic_masks],
                 total_energy=p.total_energy,selected_energy=p.selected_energy,
                 selected_directions=p.selected_directions,budget_allocation='coverage',
                 reserved_energy=p.reserved_energy,
                 lowest_group_coverage_feasible=p.lowest_group_coverage_feasible,
                 coverage_fallback_to_global=p.fallback_to_global,
                 coverage_changed_from_global=p.changed_from_global)
        result=validate_allocation_diagnostic(d,dict(lora_rank=10,sa_hoep_energy_budget=.05,
             sa_adaptive_a_eps=1e-8,sa_hoep_eigenvalue_rtol=1e-6,sa_hoep_budget_allocation='coverage'))
        probe_checks.append(dict(job=item['job'],task_id=item['task_id'],**result))
    write(ROOT/'proofs/real_state_math_check.json',dict(passed=True,scope='Six saved training states, no new accuracy evaluation',
          input_sha256=sha(MAIN/'records/allocation_coverage_probe.json'),checks=probe_checks))
    source_files=sorted((ROOT/'source').rglob('*.py'))
    source_hash=hashlib.sha256(''.join(str(p.relative_to(ROOT))+':'+sha(p)+'\n'
                              for p in source_files).encode()).hexdigest()
    jobs=[];reference_hashes={};smoke_readiness={}
    for phase in ('smoke','formal'):
        combinations=[('global',1),('global',10),('coverage',1),('coverage',10)] if phase=='smoke' else [('coverage',1),('coverage',10)]
        for allocation,multiplier in combinations:
            for dataset,seed in [('cub',1),('imagenetr',1995)]:
                control=choose(reference,dataset,'control','smoke')
                rate_ref=choose(reference,dataset,'control' if multiplier==1 else 'a_gain10','smoke')
                base=choose(reference,dataset,'a_gain10',phase)
                cfg=read(base['config']);arm=f'{allocation}_gain{multiplier}'
                name=f'{dataset}_seed{seed}_t50_{arm}_{phase}_20261023'
                out=ROOT/'outputs'/name;assert not out.exists()
                cfg.update(prefix=name,filepath=str(out)+'/',sa_p0_run_id=name,
                    sa_hoep_budget_allocation=allocation,sa_a_gain_experiment=multiplier!=1,
                    sa_a_gradient_multiplier=float(multiplier))
                config=ROOT/'configs'/(name+'.json');write(config,cfg)
                frozen=choose(parent,dataset,'frozen','formal');hoep=choose(parent,dataset,'hoep','formal')
                matched_rate=hoep if multiplier==1 else choose(reference,dataset,'a_gain10','formal')
                for ref in [control,rate_ref]:
                    hashes=Path(ref['output'])/'p0_hashes.json'
                    assert read(Path(ref['output'])/'metrics.json')['tasks']==3
                    assert set(read(hashes))=={'0','1','2'}
                    reference_hashes[str(hashes)]=sha(hashes)
                readiness_key=f'{dataset}_gain{multiplier}'
                if readiness_key not in smoke_readiness:
                    schedule=read(Path(rate_ref['output'])/'task_snapshots/task_002/hoep_partition_schedule.json')
                    changes=[]
                    for task in [1,2]:
                        values=[torch.tensor(v,dtype=torch.float64) for v in schedule['diagnostics'][str(task)]['branch_spectra']]
                        p=select_with_branch_coverage(values,.05,1e-6,1e-8)
                        changes.append(p.changed_from_global)
                    assert any(changes),('three-task check would not exercise changed allocation',readiness_key)
                    smoke_readiness[readiness_key]=changes
                signature=hashlib.sha256(json.dumps({'source':source_hash,'config':{k:v for k,v in cfg.items()
                    if k not in {'prefix','filepath','sa_p0_run_id','config'}}},sort_keys=True).encode()).hexdigest()
                jobs.append(dict(name=name,phase=phase,dataset=dataset,seed=seed,arm=arm,
                    multiplier=multiplier,allocation=allocation,config=str(config),output=str(out),
                    log=str(ROOT/'logs'/(name+'.log')),signature=signature,
                    task0_reference=control['output'],three_task_reference=rate_ref['output'],
                    matched_frozen=frozen['output'],matched_hoep=hoep['output'],matched_rate=matched_rate['output']))
    assert len(jobs)==12 and len({j['signature'] for j in jobs})==12
    write(ROOT/'proofs/smoke_readiness.json',dict(passed=True,scope='Existing same-gain training states only; not new model evidence',
          changed_allocations_at_tasks_1_2=smoke_readiness))
    manifest=dict(source=str(ROOT/'source'),source_reference=str(REFERENCE),primary=str(MAIN),
        gpus=[0,1,3,4,5,7],priority_wait_gpu=0,jobs=jobs,goal=parent['goal'],
        hypotheses={'coverage_gain1':'Minimum branch coverage inside the original 5% global energy cap',
                    'coverage_gain10':'Same allocation plus the prespecified A gradient multiplier10'},
        training_source_sha256=source_hash,reference_smoke_hashes=reference_hashes,
        intentional_core_changes=sorted(CHANGED),new_training_modules=sorted(EXTRA))
    files=source_files+sorted(ROOT.glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))
    files+=[Path(j['config']) for j in jobs]
    files+=sorted(p for p in (ROOT/'proofs').iterdir() if p.is_file())
    files+=[ROOT/'records/preparation_context.json']
    manifest['file_sha256']={str(p.relative_to(ROOT)):sha(p) for p in files}
    write(ROOT/'manifest.json',manifest)
    write(ROOT/'status.json',dict(state='prepared',jobs={j['name']:dict(state='pending',completed_tasks=0) for j in jobs}))
    print(json.dumps(dict(prepared_jobs=len(jobs),smoke=8,formal=4,training_source_sha256=source_hash,
                         source_files=len(source_files),frozen_files=len(manifest['file_sha256']),models_started=0),indent=2))


if __name__=='__main__':main()
