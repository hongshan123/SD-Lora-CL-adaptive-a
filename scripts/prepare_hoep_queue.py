"""Create local queue inputs from source-only templates without launching jobs."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, data_root, primary_root=None, gpus=None):
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    plan = json.loads((root / 'queue_plan.json').read_text())
    extension = 'formula' in plan
    if extension and primary_root is None:
        raise ValueError('B-metric preparation requires --primary-root')
    primary = Path(primary_root).resolve() if primary_root is not None else None
    for dataset in ('cub', 'imagenet-r'):
        for split in ('train', 'test'):
            if not (data_root / dataset / split).is_dir():
                raise FileNotFoundError(data_root / dataset / split)
    if (root / 'manifest.json').exists() or (root / 'status.json').exists():
        raise FileExistsError('Use a fresh experiment copy; existing queue inputs are preserved')
    devices = list(plan['gpus'] if gpus is None else gpus)
    if not devices or len(devices) != len(set(devices)) or any(g < 0 for g in devices):
        raise ValueError('Provide distinct nonnegative physical GPU IDs')
    if extension and len(devices) != len(plan['gpus']):
        raise ValueError('B-metric queue requires one GPU per configured lane')
    data_link = root / 'source/data'
    if data_link.exists() or data_link.is_symlink():
        if data_link.resolve() != data_root:
            raise ValueError('source/data already points to a different dataset')
    else:
        data_link.symlink_to(data_root, target_is_directory=True)
    for folder in ('outputs', 'logs', 'proofs', 'records', 'runtime/configs'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    source_files = sorted((root / 'source').rglob('*.py'))
    source_hash = hashlib.sha256(''.join(
        str(p.relative_to(root)) + ':' + digest(p) + '\n' for p in source_files
    ).encode()).hexdigest()
    jobs = []
    for spec in plan['jobs']:
        job = dict(spec)
        name = job['name']
        output = root / 'outputs' / name
        if output.exists():
            raise FileExistsError(output)
        config = json.loads((root / 'configs' / (name + '.json')).read_text())
        config['filepath'] = str(output) + '/'
        local_config = root / 'runtime/configs' / (name + '.json')
        local_config.write_text(json.dumps(config, indent=2) + '\n')
        effective = {k: v for k, v in config.items()
                     if k not in {'prefix', 'filepath', 'sa_p0_run_id', 'config'}}
        signature = hashlib.sha256(json.dumps(
            {'source': source_hash, 'config': effective}, sort_keys=True
        ).encode()).hexdigest()
        job.update(config=str(local_config), output=str(output),
                   log=str(root / 'logs' / (name + '.log')), signature=signature)
        if extension:
            lane = plan['gpus'].index(spec['gpu'])
            job['gpu'] = devices[lane]
            prefix = f"{job['dataset']}_seed{job['seed']}_t50_"
            for key, arm in [('matched_frozen', 'frozen'), ('matched_hoep', 'hoep')]:
                job[key] = str(primary / 'outputs' / (prefix + arm + '_formal_20261023'))
            job['task0_reference'] = str(
                primary / 'outputs' / (prefix + 'hoep_smoke_20261023'))
        jobs.append(job)
    if len({job['signature'] for job in jobs}) != len(jobs):
        raise ValueError('Duplicate effective experiment configurations')
    manifest = {key: value for key, value in plan.items() if key != 'jobs'}
    manifest.update(source=str(root / 'source'), gpus=devices, jobs=jobs,
                    training_source_sha256=source_hash)
    if extension:
        manifest['primary'] = str(primary)
    files = source_files + sorted(root.glob('*.py')) + [root / 'queue_plan.json']
    files += [Path(job['config']) for job in jobs]
    manifest['file_sha256'] = {str(p.relative_to(root)): digest(p) for p in files}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--primary-root', type=Path)
    parser.add_argument('--gpus', type=int, nargs='+')
    args = parser.parse_args()
    manifest = prepare(args.suite, args.data_root, args.primary_root, args.gpus)
    print('Prepared', len(manifest['jobs']), 'jobs; no training launched')
