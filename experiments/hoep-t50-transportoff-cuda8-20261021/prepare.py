#!/usr/bin/env python3
"""Prepare paired Transport ablations from the original immutable sources."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PROJECTS = Path('/home/hongzhijun/hongshan/SD-lora-cl_2')
SPEED = PROJECTS / 'SD-Lora-CL-hoep-normcap-speed-20261012/.runtime_hoep_normcap_speed_20261012'
GRID = PROJECTS / 'SD-Lora-CL-hoep-taskgrid-20261012/.runtime_hoep_taskgrid_no_normcap_1seed_20261012'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (ROOT / 'manifest.json').exists(), 'Refusing to replace an existing release'
    sources = {}
    for key, runtime in [('speed', SPEED), ('taskgrid', GRID)]:
        pin = json.loads((runtime / 'frozen_sources.json').read_text())
        original = runtime / 'source_snapshot'
        for relative, digest in pin['source_hashes'].items():
            assert sha(original / relative) == digest, (key, relative)
        for variant in ['reference', 'off']:
            target = ROOT / ('source_' + key + '_' + variant)
            shutil.copytree(original, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            target.joinpath('data').symlink_to(PROJECTS / 'SD-Lora-CL-hoep-normcap-speed-20261012/data', target_is_directory=True)
            if variant == 'off':
                model = target / 'models/sa_sdlora.py'
                before = model.read_text()
                old = '            if self._coordinate_stable_transport:\n                self._rebuild_eval_backbone()'
                new = ('            if (self._coordinate_stable_transport\n'
                       '                    or self.args.get("sa_rebuild_eval_every_task", False)):\n'
                       '                self._rebuild_eval_backbone()')
                assert before.count(old) == 1
                after = before.replace(old, new)
                model.write_text(after)
                (ROOT / ('patch_' + key + '.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='original/models/sa_sdlora.py', tofile='transport_off/models/sa_sdlora.py')))
            hashes = {relative: sha(target / relative) for relative in pin['source_hashes']}
            changed = [relative for relative in hashes if hashes[relative] != pin['source_hashes'][relative]]
            assert changed == ([] if variant == 'reference' else ['models/sa_sdlora.py'])
            sources[key + '_' + variant] = dict(path=str(target), original_snapshot=str(original), original_git_revision=pin['git_revision'], original_hashes=pin['source_hashes'], hashes=hashes, changed_files=changed)
    jobs = []
    specs = [
        ('imagenetr', 1995, 'speed', SPEED / 'configs/recover_hoep_imagenetr_nc0_seed1995_t50_speed_20261012.json', 76.67, 80.5732),
        ('cub', 1, 'taskgrid', GRID / 'configs/hoep_cub_transport_nc0_seed1_t50_e20_bs128_20261012.json', 81.67, 88.8572),
        ('cifar224', 1993, 'speed', SPEED / 'configs/recover_hoep_cifar224_nc0_seed1993_t50_speed_20261012.json', 85.26, 90.7908),
    ]
    for dataset, seed, key, reference_path, final, aaa in specs:
        reference = json.loads(reference_path.read_text())
        name = f'hoep_{dataset}_T50_seed{seed}_transport0_normcap0_ddp2'
        job = dict(name=name, dataset=dataset, seed=seed, tasks=50, original_config=str(reference_path), original_config_sha256=sha(reference_path), original_transport_on_final=final, original_transport_on_aaa=aaa, phases={})
        write(ROOT / 'references' / (name + '.json'), reference)
        for phase in ['smoke_on', 'smoke_off', 'formal']:
            config = dict(reference)
            run_id = name + '_' + phase + '_20261021'
            config.update(prefix=run_id, filepath=str(ROOT / 'outputs' / run_id) + '/', sa_p0_run_id=run_id)
            if phase != 'smoke_on':
                config.update(sa_coordinate_stable_transport=False, sa_rebuild_eval_every_task=True)
            if phase.startswith('smoke'):
                config.update(init_epoch=1, epochs=1, max_tasks=2)
            changed = {k: {'before': reference.get(k), 'after': config.get(k)} for k in set(reference) | set(config) if reference.get(k) != config.get(k)}
            allowed = {'prefix', 'filepath', 'sa_p0_run_id', 'sa_coordinate_stable_transport', 'sa_rebuild_eval_every_task'}
            if phase.startswith('smoke'):
                allowed.update({'init_epoch', 'epochs', 'max_tasks'})
            assert set(changed) <= allowed
            assert config['sa_live_a_absorb_mode'] == 'operator_preserving_absorb'
            assert config['sa_adaptive_a_strategy'] == 'operator_energy_partition'
            assert config['sa_hoep_energy_budget'] == .05
            assert config['batch_size'] == 64 and config['device'] == ['0', '1']
            config_path = ROOT / 'configs' / (run_id + '.json')
            write(config_path, config)
            entry = dict(name=run_id, method='hoep', dataset=dataset, seed=seed, tasks=config['max_tasks'], source=key + ('_reference' if phase == 'smoke_on' else '_off'), config=str(config_path), config_sha256=sha(config_path), output=config['filepath'].rstrip('/'), log=str(ROOT / 'logs' / (run_id + '.log')), changed_config=changed)
            job['phases'][phase] = entry
        jobs.append(job)
    for directory in ['outputs', 'logs', 'proofs']:
        (ROOT / directory).mkdir(exist_ok=True)
    write(ROOT / 'manifest.json', dict(host='cuda8', gpu_pairs=[[0, 1], [2, 3]], method='HOEP-A', tasks=50, rank=10, energy_budget=.05, normcap=False, transport=False, ls_align=True, global_batch=128, batch_per_rank=64, epochs_per_task=20, jobs=jobs, sources=sources, boundary_rebuild='Preserve the Transport-ON deployment rebuild at every boundary; only disable old-prototype transport.'))
    write(ROOT / 'status.json', dict(state='prepared', jobs={j['name']: {'state': 'pending'} for j in jobs}))
    print('PREPARED', len(jobs), 'formal runs; exact source hashes pinned; paired two-task smoke gates')


if __name__ == '__main__':
    main()
