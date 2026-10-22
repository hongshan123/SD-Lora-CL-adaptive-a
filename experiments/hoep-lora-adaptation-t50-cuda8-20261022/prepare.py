"""Generate machine-local queue inputs from tracked configuration templates."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prepare(root, data_root, wait_root):
    root, data_root, wait_root = map(lambda p: Path(p).resolve(), (root, data_root, wait_root))
    plan = json.loads((root / 'queue_plan.json').read_text())
    assert (data_root / 'cifar-100-python/train').is_file()
    assert (data_root / 'imagenet-r/train').is_dir()
    assert (data_root / 'imagenet-r/test').is_dir()
    assert (wait_root / 'status.json').is_file(), 'upstream training status is required'
    assert (wait_root / 'replay_status.json').is_file(), 'upstream replay status is required'
    if (root / 'manifest.json').exists():
        raise FileExistsError('queue already prepared; use a fresh copy for another queue')
    data_link = root / 'source/data'
    if data_link.is_symlink() or data_link.exists():
        assert data_link.resolve() == data_root, 'source/data points to a different dataset'
    else:
        data_link.symlink_to(data_root, target_is_directory=True)
    for folder in ('outputs', 'logs', 'proofs', 'runtime/configs'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    jobs = []
    for spec in plan['jobs']:
        config = json.loads((root / 'configs' / (spec['name']+'.json')).read_text())
        output = root / 'outputs' / spec['name']
        assert not output.exists(), ('fresh output required', str(output))
        config['filepath'] = str(output) + '/'
        path = root / 'runtime/configs' / (spec['name']+'.json')
        path.write_text(json.dumps(config, indent=2)+'\n')
        jobs.append(dict(spec, config=str(path), output=str(output),
                         log=str(root / 'logs' / (spec['name']+'.log'))))
    files = list((root/'source').rglob('*.py'))
    files += [root/name for name in ('run_job.py', 'run_queue.py', 'gpu_leases.py')]
    files += [Path(job['config']) for job in jobs]
    manifest = dict(protocol=plan['protocol'], source=str(root/'source'),
                    upstream_root=str(wait_root), jobs=jobs,
                    file_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(files)})
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--wait-root', type=Path, required=True,
                        help='previous training/replay directory whose success the queue must await')
    args = parser.parse_args()
    manifest = prepare(ROOT, args.data_root, args.wait_root)
    print('Prepared', len(manifest['jobs']), 'jobs. Run: python run_queue.py')
