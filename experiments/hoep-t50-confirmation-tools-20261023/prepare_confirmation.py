"""Prepare paired T50 confirmation inputs; never launch or resume training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


SUITES = {
    'main': 'hoep-t50-improvement-cuda8-20261023',
    'b_metric': 'hoep-bmetric-t50-cuda8-20261023',
    'a_step': 'hoep-a-step-t50-cuda8-20261023',
}
OPTIONAL_SUITES = {'allocation': 'hoep-budget-allocation-t50-cuda8-20261023'}
ALLOCATION_CORE = {'backbone/sa_lora.py', 'models/sa_sdlora.py', 'utils/inc_net.py'}
SEEDS = {'cub': [1, 2, 3], 'imagenetr': [1995, 1996, 1997]}
OPERATIONAL = {'prefix', 'filepath', 'config', 'sa_p0_run_id'}
METHOD = {
    'sa_train_a_all_tasks', 'sa_adaptive_a_enabled', 'sa_total_history_weight',
    'sa_allseen_proto_weight', 'sa_b_metric_kind', 'sa_b_metric_damping',
    'sa_b_metric_history_mix', 'sa_a_gain_experiment', 'sa_a_gradient_multiplier',
    'sa_hoep_budget_allocation',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(source_hash, config):
    scientific = {k: v for k, v in config.items() if k not in OPERATIONAL}
    payload = {'source': source_hash, 'config': scientific}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def verified_family(root):
    root = Path(root).resolve()
    manifest = read(root / 'manifest.json')
    files = manifest['file_sha256']
    for relative, expected in files.items():
        path = root / relative
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError(f'Frozen file changed: {path}')
    actual = {str(p.relative_to(root)) for p in (root / 'source').rglob('*.py')}
    frozen = {p for p in files if p.startswith('source/') and p.endswith('.py')}
    if actual != frozen:
        raise ValueError(f'Untracked/missing training source in {root}')
    if 'run_job.py' not in files:
        raise ValueError('The training worker must be frozen')
    return {'root': root, 'manifest': manifest}


def job_arm(job):
    return job.get('arm', 'bmetric_' + job.get('kind', ''))


def protocol(config):
    expected = {
        'batch_size': 128, 'lora_rank': 10, 'init_cls': 4, 'increment': 4,
        'max_tasks': 50, 'init_epoch': 20, 'epochs': 20,
        'optimizer': 'sgd', 'sa_live_a_absorb_mode': 'operator_preserving_absorb',
        'sa_live_a_coordinate_align': True, 'sa_coordinate_stable_transport': True,
        'sa_boundary_snapshots': True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Protocol differs at {key}: {config.get(key)!r}')
    if config.get('sa_resume') or config.get('distributed'):
        raise ValueError('Fresh single-GPU training is required')
    if len(config['device']) != 1:
        raise ValueError('Single-GPU batch128 is required')
    if config.get('sa_gradient_accumulation_steps', 1) != 1:
        raise ValueError('Gradient accumulation changes this protocol')


def existing_index(base):
    """Find configuration duplicates without treating status as proof of life."""
    index = {}
    for path in sorted(Path(base).glob('*/manifest.json')):
        manifest = read(path)
        if not isinstance(manifest.get('jobs'), list):
            continue
        status_path = path.parent / 'status.json'
        states = read(status_path).get('jobs', {}) if status_path.exists() else {}
        for job in manifest['jobs']:
            source = job.get('training_source_sha256', manifest.get('training_source_sha256'))
            if not source:
                continue
            config_path = Path(job.get('config', ''))
            if not config_path.is_file():
                continue
            config = read(config_path)
            digest = signature(source, config)
            index.setdefault(digest, []).append({
                'suite': str(path.parent), 'name': job.get('name'),
                'config': str(config_path), 'config_sha256': sha(config_path),
                'output': job.get('output'),
                'observed_state': states.get(job.get('name'), {}).get('state'),
                'state_independently_verified': False,
            })
    return index


def prepare(base, destination, arms):
    base, destination = Path(base).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not arms or len(set(arms)) != len(arms):
        raise ValueError('Provide distinct candidate arms')
    families = {key: verified_family(base / name) for key, name in SUITES.items()}
    for key, name in OPTIONAL_SUITES.items():
        if (base / name / 'manifest.json').exists():
            families[key] = verified_family(base / name)
    main = families['main']['manifest']
    if main['goal']['confirmation_seeds'] != SEEDS:
        raise ValueError('Prespecified confirmation seeds changed')
    # Added optional modules and factory routes can differ; the shared core cannot.
    for key, family in families.items():
        if key == 'main':
            continue
        core = families['a_step' if key == 'allocation' else 'main']['root'] / 'source'
        if key == 'allocation':
            if family['manifest']['source_reference'] != str(families['a_step']['root']):
                raise ValueError('Unknown allocation source reference')
            reference_files = {p.relative_to(core) for p in core.rglob('*.py')}
            actual_files = {p.relative_to(family['root'] / 'source')
                            for p in (family['root'] / 'source').rglob('*.py')}
            if ({str(p) for p in actual_files-reference_files} != {'backbone/budget_allocation.py'}
                    or reference_files-actual_files):
                raise ValueError('Unexpected allocation source files')
        for path in core.rglob('*.py'):
            relative = path.relative_to(core)
            allowed = ALLOCATION_CORE if key == 'allocation' else {'utils/factory.py'}
            if str(relative) not in allowed:
                if path.read_bytes() != (family['root'] / 'source' / relative).read_bytes():
                    raise ValueError(f'Unmatched shared training core: {key}/{relative}')
    catalogue = {}
    for key, family in families.items():
        for job in family['manifest']['jobs']:
            if job['phase'] != 'formal':
                continue
            item = (job['dataset'], job_arm(job))
            if item in catalogue:
                raise ValueError(f'Duplicate formal template: {item}')
            config = read(job['config'])
            protocol(config)
            catalogue[item] = (key, job, config)
    for arm in arms:
        if arm in ('frozen', 'hoep'):
            raise ValueError('Choose a candidate; Frozen controls are added once')
        for dataset in SEEDS:
            if (dataset, arm) not in catalogue:
                raise ValueError(f'No prespecified template for {dataset}/{arm}')
            frozen = catalogue[dataset, 'frozen'][2]
            candidate = catalogue[dataset, arm][2]
            differences = {k for k in frozen.keys() | candidate.keys()
                           if frozen.get(k) != candidate.get(k)}
            if differences - OPERATIONAL - METHOD:
                raise ValueError(f'Unmatched pair: {dataset}/{arm}/{differences}')
            if frozen['sa_train_a_all_tasks'] or frozen['sa_adaptive_a_enabled']:
                raise ValueError('Frozen control is trainable')
            if not candidate['sa_train_a_all_tasks'] or not candidate['sa_adaptive_a_enabled']:
                raise ValueError('Candidate does not train A with HOEP')
    index = existing_index(base)
    jobs, references, configs = [], [], {}
    for dataset, seeds in SEEDS.items():
        for arm in ['frozen'] + list(arms):
            family_key, template, config = catalogue[dataset, arm]
            family = families[family_key]
            if template['seed'] != seeds[0] or config['seed'] != [seeds[0]]:
                raise ValueError('Initial template is not the prespecified first seed')
            source_hash = family['manifest']['training_source_sha256']
            for seed in seeds:
                copied = dict(config)
                name = f'{dataset}_seed{seed}_t50_{arm}_confirmation'
                copied.update(seed=[seed], prefix=name, sa_p0_run_id=name,
                              filepath=str(destination / 'outputs' / name) + '/')
                copied.pop('config', None)
                if 'sa_hoep_partition_seed' in copied:
                    copied['sa_hoep_partition_seed'] = seed
                digest = signature(source_hash, copied)
                prior = index.get(digest, [])
                if seed == seeds[0]:
                    # Preserve the original partition seed for exact template reuse.
                    if signature(source_hash, config) != digest:
                        raise ValueError('First-seed configuration would change')
                    if not prior:
                        raise ValueError('Initial seed reference is missing')
                if prior:
                    references.append({'dataset': dataset, 'seed': seed, 'arm': arm,
                                       'signature': digest, 'existing': prior})
                    continue
                local_config = destination / 'configs' / (name + '.json')
                jobs.append({
                    'name': name, 'phase': 'formal', 'dataset': dataset,
                    'seed': seed, 'arm': arm, 'source_family': family_key,
                    'source': str(family['root'] / 'source'),
                    'worker': str(family['root'] / 'run_job.py'),
                    'config': str(local_config), 'output': copied['filepath'],
                    'log': str(destination / 'logs' / (name + '.log')),
                    'training_source_sha256': source_hash, 'signature': digest,
                })
                configs[local_config.name] = copied
    if len({job['signature'] for job in jobs}) != len(jobs):
        raise ValueError('Duplicate proposed training jobs')
    plan = {
        'preparation_only': True, 'training_started': False,
        'goal': main['goal'], 'arms': list(arms), 'jobs': jobs,
        'existing_references': references,
        'reference_policy': 'References require process/result audit before use; failed or stale runs are preserved, never silently rerun.',
        'families': {key: {
            'root': str(family['root']),
            'training_source_sha256': family['manifest']['training_source_sha256'],
            'file_sha256': family['manifest']['file_sha256'],
        } for key, family in families.items()},
        'seed_policy': 'All prespecified seeds, no score-dependent omission. First seeds are reused, Frozen controls shared across arms.',
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.confirmation-prepare-', dir=destination.parent))
    try:
        (staging / 'configs').mkdir()
        for name, config in configs.items():
            (staging / 'configs' / name).write_text(json.dumps(config, indent=2) + '\n')
        plan['config_sha256'] = {name: sha(staging / 'configs' / name) for name in configs}
        (staging / 'manifest.json').write_text(json.dumps(plan, indent=2) + '\n')
        # The local plan is a fresh input artifact; no output directory is created.
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', required=True)
    args = parser.parse_args()
    result = prepare(args.base, args.destination, args.arms)
    print(f"Prepared {len(result['jobs'])} fresh job inputs; "
          f"{len(result['existing_references'])} existing references. No training launched.")
