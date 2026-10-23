"""Activate only the four already-planned additional-seed Frozen controls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN = ROOT.parent / 'hoep-t50-improvement-cuda8-20261023'
INPUTS = ROOT.parent / 'hoep-t50-confirmation-inputs-cuda8-20261023'
OPERATIONAL = {'prefix', 'filepath', 'config', 'sa_p0_run_id'}
SEEDS = {'cub': [2, 3], 'imagenetr': [1996, 1997]}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def signature(source, config):
    scientific = {k: v for k, v in config.items() if k not in OPERATIONAL}
    return hashlib.sha256(json.dumps({'source': source, 'config': scientific},
                                    sort_keys=True).encode()).hexdigest()


def main():
    if (ROOT / 'manifest.json').exists() or (ROOT / 'status.json').exists():
        raise FileExistsError('Execution inputs already exist; never overwrite them')
    parent, inputs = read(MAIN / 'manifest.json'), read(INPUTS / 'manifest.json')
    assert inputs['preparation_only'] is True and inputs['training_started'] is False
    assert parent['goal']['confirmation_seeds'] == {'cub': [1, 2, 3], 'imagenetr': [1995, 1996, 1997]}
    for relative, digest in parent['file_sha256'].items():
        assert sha(MAIN / relative) == digest, relative
    assert sha(ROOT / 'run_job.py') == sha(MAIN / 'run_job.py')
    old = (MAIN / 'run_queue.py').read_text()
    expected = old.replace("for phase in ('smoke','formal'):phase_run(phase,manifest,status)",
                           "for phase in ('formal',):phase_run(phase,manifest,status)")
    assert (ROOT / 'run_queue.py').read_text() == expected
    gates = read(MAIN / 'proofs/smoke_equality.json')
    for dataset in ('cub', 'imagenetr'):
        assert gates[dataset]['common_task0_all_arms']
        assert gates[dataset]['disabled_learner_three_tasks_exact']
    declared = {key: value for key, value in parent['file_sha256'].items()
                if key.startswith('source/') and key.endswith('.py')}
    actual = {'source/' + str(p.relative_to(MAIN / 'source'))
              for p in (MAIN / 'source').rglob('*.py')}
    assert set(declared) == actual
    for dataset in ('cub', 'imagenet-r'):
        for split in ('train', 'test'):
            assert (MAIN / 'source/data' / dataset / split).is_dir()
    link = ROOT / 'source'
    if link.exists() or link.is_symlink():
        assert link.resolve() == (MAIN / 'source').resolve()
    else:
        link.symlink_to(MAIN / 'source', target_is_directory=True)
    for folder in ('configs', 'outputs', 'logs', 'proofs', 'records'):
        (ROOT / folder).mkdir(exist_ok=True)
    planned = [j for j in inputs['jobs'] if j['arm'] == 'frozen']
    assert {(j['dataset'], j['seed']) for j in planned} == {
        (dataset, seed) for dataset, seeds in SEEDS.items() for seed in seeds}
    assert len(planned) == 4
    source = parent['training_source_sha256']
    existing = []
    for path in ROOT.parent.glob('*/manifest.json'):
        value = read(path)
        if value.get('preparation_only'):
            continue
        existing += [j for j in value.get('jobs', []) if isinstance(j, dict)]
    jobs, activation = [], []
    # Alternate datasets; this order is independent of all test scores.
    for dataset, seed in [('cub', 2), ('imagenetr', 1996), ('cub', 3), ('imagenetr', 1997)]:
        job = next(j for j in planned if (j['dataset'], j['seed']) == (dataset, seed))
        assert job['source_family'] == 'main' and job['training_source_sha256'] == source
        original = Path(job['config'])
        assert sha(original) == inputs['config_sha256'][original.name]
        config = read(original)
        first = next(j for j in parent['jobs'] if j['phase'] == 'formal'
                     and j['dataset'] == dataset and j['arm'] == 'frozen')
        template = read(first['config'])
        differences = {key for key in config.keys() | template.keys()
                       if config.get(key) != template.get(key)}
        assert differences <= OPERATIONAL | {'seed', 'sa_hoep_partition_seed'}, differences
        assert config['seed'] == [seed] and config['sa_hoep_partition_seed'] == seed
        assert config['sa_train_a_all_tasks'] is False and config['sa_adaptive_a_enabled'] is False
        assert signature(source, config) == job['signature']
        assert not any(j.get('signature') == job['signature'] for j in existing), 'Already activated'
        assert not Path(job['output']).exists(), 'Planned input unexpectedly has output'
        copied = dict(config)
        output = ROOT / 'outputs' / job['name']
        assert not output.exists()
        copied['filepath'] = str(output) + '/'
        assert signature(source, copied) == job['signature']
        destination = ROOT / 'configs' / original.name
        assert not destination.exists()
        write(destination, copied)
        jobs.append(dict(name=job['name'], phase='formal', dataset=dataset, seed=seed,
                         arm='frozen', source_family='main', config=str(destination),
                         output=str(output), log=str(ROOT / 'logs' / (job['name'] + '.log')),
                         signature=job['signature'], training_source_sha256=source))
        activation.append(dict(name=job['name'], input_config=str(original),
                               input_sha256=sha(original), config_sha256=sha(destination),
                               signature=job['signature'], scientific_config_unchanged=True))
    files = {key: digest for key, digest in declared.items()}
    files.update({p.name: sha(p) for p in ROOT.glob('*.py')})
    files.update({str(Path(j['config']).relative_to(ROOT)): sha(j['config']) for j in jobs})
    manifest = dict(source=str(ROOT / 'source'), source_reference=str(MAIN),
                    training_source_sha256=source, gpus=[6], jobs=jobs,
                    goal=parent['goal'], file_sha256=files,
                    preparation_only=False, activates_existing_planned_inputs=True,
                    preparation_manifest=str(INPUTS / 'manifest.json'),
                    preparation_manifest_sha256=sha(INPUTS / 'manifest.json'),
                    source_reference_manifest_sha256=sha(MAIN / 'manifest.json'))
    write(ROOT / 'proofs/input_activation.json', dict(training_started=False, jobs=activation,
          worker_byte_identical=True, scheduler_change='Omit smoke phase for unchanged Frozen controls',
          original_smoke_proof_sha256=sha(MAIN / 'proofs/smoke_equality.json')))
    write(ROOT / 'manifest.json', manifest)
    print(json.dumps(dict(jobs=len(jobs), gpus=[6], no_models_started=True,
                          source_sha256=source), indent=2))


if __name__ == '__main__':
    main()
