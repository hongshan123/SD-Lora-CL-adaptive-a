"""Freeze fresh averaging inputs; no training is started by preparation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
MAIN=ROOT.parent/'hoep-t50-improvement-cuda8-20261023'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    if (ROOT/'manifest.json').exists() or (ROOT/'status.json').exists():
        raise FileExistsError('Frozen queue inputs already exist')
    parent=read(MAIN/'manifest.json')
    for path,digest in parent['file_sha256'].items():assert sha(MAIN/path)==digest,path
    expected={p.relative_to(MAIN/'source') for p in (MAIN/'source').rglob('*.py')}
    actual={p.relative_to(ROOT/'source') for p in (ROOT/'source').rglob('*.py')}
    assert {str(p) for p in actual-expected}=={'backbone/branch_operator_average.py','models/hoep_operator_average.py'}
    changed={str(p) for p in expected if (ROOT/'source'/p).read_bytes()!=(MAIN/'source'/p).read_bytes()}
    assert changed=={'utils/factory.py'}
    for p in list((ROOT/'source').rglob('*.py'))+list(ROOT.glob('*.py')):ast.parse(p.read_bytes(),filename=str(p))
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    text=result.stdout+result.stderr;(ROOT/'proofs/unit_tests.log').write_text(text)
    assert result.returncode==0 and re.search(r'Ran 9 tests',text),text
    write(ROOT/'proofs/unit_tests_passed.json',dict(passed=True,tests=9,log_sha256=sha(ROOT/'proofs/unit_tests.log')))
    write(ROOT/'proofs/source_scope.json',dict(reference=str(MAIN),changed=sorted(changed),
          added=sorted(str(p) for p in actual-expected),unchanged_files=len(expected)-len(changed)))
    data=(MAIN/'source/data').resolve()
    for dataset in ('cub','imagenet-r'):
        for split in ('train','test'):assert (data/dataset/split).is_dir()
    (ROOT/'source/data').symlink_to(data,target_is_directory=True)
    source_files=sorted((ROOT/'source').rglob('*.py'))
    source_hash=hashlib.sha256(''.join(str(p.relative_to(ROOT))+':'+sha(p)+'\n' for p in source_files).encode()).hexdigest()
    jobs=[];refs={}
    for phase in ('smoke','formal'):
        for arm in (('control','operator_average') if phase=='smoke' else ('operator_average',)):
            for dataset,seed in [('cub',1),('imagenetr',1995)]:
                base=next(j for j in parent['jobs'] if j['phase']==phase and j['dataset']==dataset and j['arm']=='hoep')
                smoke=next(j for j in parent['jobs'] if j['phase']=='smoke' and j['dataset']==dataset and j['arm']=='hoep')
                frozen=next(j for j in parent['jobs'] if j['phase']=='formal' and j['dataset']==dataset and j['arm']=='frozen')
                current=next(j for j in parent['jobs'] if j['phase']=='formal' and j['dataset']==dataset and j['arm']=='hoep')
                hashes=Path(smoke['output'])/'p0_hashes.json'
                assert set(read(hashes))=={'0','1','2'} and read(Path(smoke['output'])/'metrics.json')['tasks']==3
                refs[str(hashes)]=sha(hashes)
                cfg=copy.deepcopy(read(base['config']))
                assert cfg['sa_live_a_absorb_mode']=='operator_preserving_absorb' and cfg['batch_size']==128
                assert cfg['sa_total_history_weight']==cfg['sa_allseen_proto_weight']==0
                name=f'{dataset}_seed{seed}_t50_{arm}_{phase}_20261023';out=ROOT/'outputs'/name
                assert not out.exists()
                cfg.update(prefix=name,filepath=str(out)+'/',sa_p0_run_id=name,
                           sa_current_branch_average=arm=='operator_average',sa_branch_average_fraction=.25)
                config=ROOT/'configs'/(name+'.json');write(config,cfg)
                effective={k:v for k,v in cfg.items() if k not in {'prefix','filepath','config','sa_p0_run_id'}}
                signature=hashlib.sha256(json.dumps(dict(source=source_hash,config=effective),sort_keys=True).encode()).hexdigest()
                jobs.append(dict(name=name,phase=phase,dataset=dataset,seed=seed,arm=arm,config=str(config),output=str(out),
                                 log=str(ROOT/'logs'/(name+'.log')),signature=signature,
                                 task0_reference=smoke['output'],three_task_reference=smoke['output'],
                                 matched_frozen=frozen['output'],matched_hoep=current['output']))
    assert len(jobs)==6 and len({j['signature'] for j in jobs})==6
    files=source_files+sorted(ROOT.glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))
    files+=[Path(j['config']) for j in jobs]+[ROOT/'proofs/unit_tests_passed.json',ROOT/'proofs/source_scope.json',ROOT/'proofs/unit_tests.log']
    manifest=dict(source=str(ROOT/'source'),source_reference=str(MAIN),gpus=[0,1,3,4,5,7],
                  goal=parent['goal'],jobs=jobs,training_source_sha256=source_hash,
                  hypothesis='Average current sBA over the final quarter of updates, project onto final A, preserve history and exact absorption',
                  reference_smoke_hashes=refs,file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    write(ROOT/'manifest.json',manifest)
    print(json.dumps(dict(jobs=6,smokes=4,formals=2,training_source_sha256=source_hash,
                          source_files=len(source_files),frozen_files=len(files),no_models_started=True)))


if __name__=='__main__':main()
