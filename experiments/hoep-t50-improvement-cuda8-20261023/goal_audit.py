"""Audit paired full-horizon evidence for the user goal; never select a seed."""
import hashlib
import json
from pathlib import Path

from journal import write_json
from run_job import parse_curve

ROOT=Path(__file__).resolve().parent
OPERATIONAL={'prefix','filepath','sa_p0_run_id','config'}
METHOD={'sa_train_a_all_tasks','sa_adaptive_a_enabled','sa_total_history_weight',
        'sa_allseen_proto_weight','sa_b_metric_kind','sa_b_metric_damping','sa_b_metric_history_mix'}


def read(path):return json.loads(Path(path).read_text())


def validated_score(root,manifest,job,record):
    if record['state']!='complete':return None
    config_path=Path(job['config']);relative=str(config_path.relative_to(root))
    assert hashlib.sha256(config_path.read_bytes()).hexdigest()==manifest['file_sha256'][relative]
    config=read(config_path);out=Path(job['output']);metrics=read(out/'metrics.json')
    assert config['max_tasks']==metrics['tasks']==50
    assert config['init_epoch']==config['epochs']==20
    assert config['batch_size']==metrics['batch_size']==128 and metrics['world_size']==1
    assert config['lora_rank']==10 and config['init_cls']==config['increment']==4
    assert config['sa_live_a_absorb_mode']=='operator_preserving_absorb' and not metrics['normcap']
    assert config['sa_coordinate_stable_transport'] and metrics['transport']
    assert metrics['dataset']==job['dataset'] and metrics['seed']==job['seed']
    assert metrics['prefix']==job['name'] and metrics['state']=='complete'
    assert len(metrics['top1'])==50 and metrics['top1'][-1]==metrics['final_top1']
    assert parse_curve(Path(metrics['training_log']).read_text(errors='replace'))==metrics['top1']
    hashes=read(out/'p0_hashes.json');assert set(hashes)==set(map(str,range(50)))
    assert len(metrics['snapshot_marker_sha256'])==50
    for task,digest in enumerate(metrics['snapshot_marker_sha256']):
        marker=out/'task_snapshots'/f'task_{task:03d}'/'complete.json'
        assert hashlib.sha256(marker.read_bytes()).hexdigest()==digest
        assert read(marker)['task_id']==task
    # Recheck the final deployed state, rather than trusting a stale status flag.
    final=out/'task_snapshots/task_049';marker=read(final/'complete.json')
    for name,digest in marker['sha256'].items():
        assert hashlib.sha256((final/name).read_bytes()).hexdigest()==digest,(job['name'],name)
    if job['arm']=='frozen':assert metrics['frozen_a_verified'] is True
    if config.get('sa_b_metric_kind','off')!='off':
        assert len(metrics['b_metric_diagnostics'])==50
        for task,record in enumerate(metrics['b_metric_diagnostics']):
            assert read(out/'b_metric'/f'task_{task:03d}.json')==record
            if task:assert record['applied_branches']>0 and record['max_relative_norm_error']<1e-5
    return config,metrics


def audit(root=ROOT):
    root=Path(root);manifest=read(root/'manifest.json');status=read(root/'status.json')
    completed={};incomplete=[]
    families=[(root,manifest,status,manifest['jobs'])]
    extra=root.parent/'hoep-bmetric-t50-cuda8-20261023'
    if (extra/'status.json').exists():
        other=read(extra/'manifest.json')
        # Both optimizers share the same training loop, data, backbone and
        # boundaries. Only the optional factory route and added modules differ.
        for path in (root/'source').rglob('*.py'):
            relative=path.relative_to(root/'source')
            if str(relative)=='utils/factory.py':continue
            assert path.read_bytes()==(extra/'source'/relative).read_bytes(),('shared core differs',str(relative))
        jobs=[dict(j,arm='bmetric_'+j['kind']) for j in other['jobs']]
        families.append((extra,other,read(extra/'status.json'),jobs))
    all_jobs=[]
    for family_root,family_manifest,family_status,jobs in families:
        for job in jobs:
            if job['phase']!='formal':continue
            all_jobs.append(job)
            value=validated_score(family_root,family_manifest,job,family_status['jobs'][job['name']])
            if value is None:incomplete.append(job['name'])
            else:completed[(job['dataset'],job['seed'],job['arm'])]=(job,*value)
    arms=sorted(set(j['arm'] for j in all_jobs if j['arm'] not in ('frozen','hoep')))
    seeds=manifest['goal']['confirmation_seeds'];threshold=manifest['goal']['required_margin_pp']
    candidates={}
    for arm in arms:
        datasets={}
        for dataset in manifest['goal']['datasets']:
            rows=[];missing=[]
            for seed in seeds[dataset]:
                frozen=completed.get((dataset,seed,'frozen'));candidate=completed.get((dataset,seed,arm))
                if frozen is None or candidate is None:missing.append(seed);continue
                fjob,fcfg,fmet=frozen;cjob,ccfg,cmet=candidate
                differences={key for key in set(fcfg)|set(ccfg) if fcfg.get(key)!=ccfg.get(key)}
                assert differences <= OPERATIONAL|METHOD, ('unmatched comparison',dataset,seed,arm,differences)
                assert fcfg['sa_train_a_all_tasks'] is False and fcfg['sa_adaptive_a_enabled'] is False
                assert ccfg['sa_train_a_all_tasks'] is True and ccfg['sa_adaptive_a_enabled'] is True
                # Common Task0 protects against inadvertently changing initialization.
                fh=read(Path(fjob['output'])/'p0_hashes.json')['0'];ch=read(Path(cjob['output'])/'p0_hashes.json')['0']
                for key in ['post_train_model_hash','post_train_rng_hash','post_eval_model_hash','post_eval_rng_hash']:
                    assert fh[key]==ch[key],('Task0 changed',dataset,seed,arm,key)
                current=completed.get((dataset,seed,'hoep'))
                rows.append(dict(seed=seed,frozen_final=fmet['final_top1'],candidate_final=cmet['final_top1'],
                                 difference_pp=cmet['final_top1']-fmet['final_top1'],
                                 current_hoep_final=None if current is None else current[2]['final_top1'],
                                 difference_to_current_hoep_pp=None if current is None else cmet['final_top1']-current[2]['final_top1'],
                                 frozen_AAA=fmet['average_accuracy'],candidate_AAA=cmet['average_accuracy']))
            mean=None if not rows else sum(row['difference_pp'] for row in rows)/len(rows)
            datasets[dataset]=dict(pairs=rows,missing_seeds=missing,mean_difference_pp=mean,
                                   confirmed_pass=not missing and mean is not None and mean>threshold)
        candidates[arm]=dict(datasets=datasets,
                             confirmed_pass=all(item['confirmed_pass'] for item in datasets.values()))
    result=dict(quantitative_target_verified=any(c['confirmed_pass'] for c in candidates.values()),
                criterion='same method; both datasets; all prespecified paired seeds; mean Final difference > 0.5pp',
                threshold_pp=threshold,candidates=candidates,incomplete_jobs=incomplete)
    write_json(root/'records/goal_audit.json',result)
    return result


if __name__=='__main__':
    result=audit()
    print(json.dumps(result,indent=2,allow_nan=False))
