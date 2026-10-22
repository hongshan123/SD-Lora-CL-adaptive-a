"""Read actual live handles and compare only matching completed task boundaries."""
import json
import ast
from pathlib import Path
import re
import time

from run_job import parse_curve,NUMBER

ROOT=Path(__file__).resolve().parent


def collect():
    roots=[ROOT,ROOT.parent/'hoep-bmetric-t50-cuda8-20261023']
    a_step=ROOT.parent/'hoep-a-step-t50-cuda8-20261023'
    if (a_step/'manifest.json').exists() and (a_step/'status.json').exists():
        roots.append(a_step)
    progress=[];curves={};groups={};entries=[]
    for root in roots:
        manifest=json.loads((root/'manifest.json').read_text())
        status=json.loads((root/'status.json').read_text())
        for job in manifest['jobs']:
            if job['phase']!='formal':continue
            state=status['jobs'][job['name']]
            if state['state'] not in ('running','complete'):continue
            count=len(list((Path(job['output'])/'task_snapshots').glob('task_*/complete.json')))
            if state['state']=='running':
                proc=Path('/proc')/str(state['pid'])
                fields=(proc/'stat').read_text().split() if proc.exists() else []
                identity=bool(fields) and fields[21]==state.get('proc_start_ticks')
                progress.append(dict(job=job['name'],gpu=state.get('gpu'),pid=state['pid'],
                                     process_state=fields[2] if fields else 'gone',
                                     identity_matches=identity,completed_tasks=count))
            path=Path(job['log'])
            if not path.exists():continue
            text=path.read_text(errors='replace')
            try:curve=parse_curve(text)
            except (ValueError,AssertionError):continue
            # Deployment snapshots precede evaluation: neither alone establishes
            # a finished evaluated boundary. Require both artifacts to exist.
            curve=curve[:count]
            method=job.get('arm','bmetric_'+job.get('kind',''))
            curves[(job['dataset'],job['seed'],method)]=curve
            rows=[]
            for raw in re.findall(r'CNN: (\{[^\n]+\})',text):
                clean=re.sub(r'np\.float(?:32|64)\(\s*('+NUMBER+r')\s*\)',r'\1',raw)
                try:rows.append(ast.literal_eval(clean))
                except (ValueError,SyntaxError):rows.append({})
            groups[(job['dataset'],job['seed'],method)]=rows
            entries.append((job,method,curve))
    pairs=[]
    for job,method,curve in entries:
        if method=='frozen':continue
        reference=curves.get((job['dataset'],job['seed'],'frozen'),[])
        common=min(len(reference),len(curve))
        if not common:continue
        fg=groups.get((job['dataset'],job['seed'],'frozen'),[])
        cg=groups.get((job['dataset'],job['seed'],method),[])
        old_f=fg[common-1].get('old') if common>1 and len(fg)>=common else None
        old_c=cg[common-1].get('old') if common>1 and len(cg)>=common else None
        new_f=fg[common-1].get('new') if len(fg)>=common else None
        new_c=cg[common-1].get('new') if len(cg)>=common else None
        pairs.append(dict(dataset=job['dataset'],seed=job['seed'],method=method,
                          completed_tasks=common,seen_classes=4*common,
                          frozen_top1=reference[common-1],candidate_top1=curve[common-1],
                          difference_pp=curve[common-1]-reference[common-1],
                          frozen_old_top1=old_f,candidate_old_top1=old_c,
                          frozen_new_top1=new_f,candidate_new_top1=new_c,
                          complete_horizon=common==50))
    result=dict(updated=time.time(),progress=progress,paired_interim=pairs,
                interpretation='same seed, seen class set and completed boundary; exploratory trajectory, not T50 evidence before 50 tasks')
    out=ROOT/'records/interim_progress.json';tmp=out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    return result


if __name__=='__main__':print(json.dumps(collect(),allow_nan=False))
