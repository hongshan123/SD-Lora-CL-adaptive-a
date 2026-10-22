"""CPU-only integration checks for protocol preservation and duplicate avoidance."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import prepare_confirmation as tool


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class ConfirmationInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.destination = self.base / 'confirmation'
        self.families = {}
        for family, name in tool.SUITES.items():
            root = self.base / name
            (root / 'source/utils').mkdir(parents=True)
            (root / 'source/common.py').write_text('VALUE = 7\n')
            (root / 'source/utils/factory.py').write_text(f'# {family} optional route\n')
            (root / 'run_job.py').write_text('# Worker fixture; no training implementation\n')
            jobs = []
            arms = {
                'main': ['frozen', 'hoep', 'history', 'prototype', 'combined'],
                'b_metric': ['bmetric_current', 'bmetric_history'],
                'a_step': ['a_gain10', 'a_gain100'],
            }[family]
            for dataset, seeds in tool.SEEDS.items():
                for arm in arms:
                    seed = seeds[0]
                    name = f'{dataset}_seed{seed}_t50_{arm}_formal'
                    config = dict(
                        dataset=dataset, seed=[seed], prefix=name, sa_p0_run_id=name,
                        filepath=str(root / 'outputs' / name) + '/', device=['0'],
                        batch_size=128, lora_rank=10, init_cls=4, increment=4,
                        max_tasks=50, init_epoch=20, epochs=20, optimizer='sgd',
                        lrate=.01, init_lr=.01, sa_resume=False,
                        sa_live_a_absorb_mode='operator_preserving_absorb',
                        sa_live_a_coordinate_align=True, sa_coordinate_stable_transport=True,
                        sa_boundary_snapshots=True, sa_train_a_all_tasks=arm != 'frozen',
                        sa_adaptive_a_enabled=arm != 'frozen',
                        sa_hoep_partition_seed=seed,
                        sa_total_history_weight=.001 if arm in ('history', 'combined') else 0.,
                        sa_allseen_proto_weight=.05 if arm in ('prototype', 'combined') else 0.,
                    )
                    if family == 'a_step':
                        config.update(sa_a_gain_experiment=True,
                                      sa_a_gradient_multiplier=10. if arm == 'a_gain10' else 100.)
                    if family == 'b_metric':
                        config.update(sa_b_metric_kind=arm.removeprefix('bmetric_'),
                                      sa_b_metric_damping=.1, sa_b_metric_history_mix=1.)
                    path = root / 'configs' / (name + '.json')
                    write(path, config)
                    job = dict(name=name, phase='formal', dataset=dataset, seed=seed,
                               config=str(path), output=config['filepath'], arm=arm)
                    if family == 'b_metric':
                        job['kind'] = job.pop('arm').removeprefix('bmetric_')
                    jobs.append(job)
            manifest = {
                'source': str(root / 'source'),
                'training_source_sha256': hashlib.sha256(family.encode()).hexdigest(),
                'file_sha256': {}, 'jobs': jobs,
                'goal': {'confirmation_seeds': copy.deepcopy(tool.SEEDS),
                         'datasets': list(tool.SEEDS), 'tasks': 50,
                         'primary_metric': 'final_top1', 'required_margin_pp': .5},
            }
            write(root / 'manifest.json', manifest)
            write(root / 'status.json', {'jobs': {j['name']: {'state': 'pending'} for j in jobs}})
            self.families[family] = root
            self.refresh(family)

    def refresh(self, family):
        root = self.families[family]
        manifest = tool.read(root / 'manifest.json')
        paths = list((root / 'source').rglob('*.py')) + list((root / 'configs').glob('*.json'))
        paths.append(root / 'run_job.py')
        manifest['file_sha256'] = {str(p.relative_to(root)): tool.sha(p) for p in paths}
        write(root / 'manifest.json', manifest)

    def test_preserves_science_and_all_seeds_without_starting_training(self):
        before = {str(p): p.read_bytes() for p in self.base.rglob('*') if p.is_file()}
        plan = tool.prepare(self.base, self.destination, ['a_gain10'])
        self.assertEqual(len(plan['jobs']), 8)
        self.assertEqual(len(plan['existing_references']), 4)
        self.assertFalse(plan['training_started'])
        self.assertFalse((self.destination / 'outputs').exists())
        self.assertFalse((self.destination / 'status.json').exists())
        self.assertEqual({(j['dataset'], j['seed']) for j in plan['jobs']},
                         {('cub', 2), ('cub', 3), ('imagenetr', 1996), ('imagenetr', 1997)})
        for job in plan['jobs']:
            cfg = tool.read(job['config'])
            family = tool.read(self.families[job['source_family']] / 'manifest.json')
            template = next(j for j in family['jobs'] if j['dataset'] == job['dataset']
                            and tool.job_arm(j) == job['arm'])
            original = tool.read(template['config'])
            changed = {k for k in cfg.keys() | original.keys() if cfg.get(k) != original.get(k)}
            self.assertEqual(changed, tool.OPERATIONAL - {'config'} | {'seed', 'sa_hoep_partition_seed'})
            self.assertEqual(cfg['seed'], [job['seed']])
            self.assertEqual(cfg['sa_hoep_partition_seed'], job['seed'])
            self.assertEqual(plan['config_sha256'][Path(job['config']).name], tool.sha(job['config']))
        self.assertTrue(all(Path(p).read_bytes() == value for p, value in before.items()))

    def test_controls_shared_across_candidates(self):
        plan = tool.prepare(self.base, self.destination, ['a_gain10', 'bmetric_current'])
        self.assertEqual(len(plan['jobs']), 12)
        self.assertEqual(sum(j['arm'] == 'frozen' for j in plan['jobs']), 4)
        self.assertEqual(len({j['signature'] for j in plan['jobs']}), 12)

    def test_existing_planned_jobs_are_not_created_again(self):
        first = tool.prepare(self.base, self.destination, ['a_gain10'])
        second = tool.prepare(self.base, self.base / 'second', ['a_gain10'])
        self.assertEqual(len(first['jobs']), 8)
        self.assertEqual(second['jobs'], [])
        self.assertEqual(len(second['existing_references']), 12)

    def test_failed_and_running_records_do_not_authorize_retraining(self):
        first = tool.prepare(self.base, self.destination, ['a_gain10'])
        statuses = {j['name']: {'state': 'failed' if j['seed'] == 2 else 'running'}
                    for j in first['jobs']}
        write(self.destination / 'status.json', {'jobs': statuses})
        second = tool.prepare(self.base, self.base / 'second', ['a_gain10'])
        self.assertFalse(second['jobs'])
        states = [ref for row in second['existing_references'] for ref in row['existing']]
        self.assertTrue(any(row['observed_state'] == 'failed' for row in states))
        self.assertTrue(any(row['observed_state'] == 'running' for row in states))
        self.assertTrue(all(not row['state_independently_verified'] for row in states))

    def test_refuses_nonpaired_learning_rate(self):
        root = self.families['a_step']
        job = tool.read(root / 'manifest.json')['jobs'][0]
        cfg = tool.read(job['config']); cfg['lrate'] = .1
        write(Path(job['config']), cfg); self.refresh('a_step')
        with self.assertRaisesRegex(ValueError, 'Unmatched pair'):
            tool.prepare(self.base, self.destination, ['a_gain10'])
        self.assertFalse(self.destination.exists())

    def test_refuses_unfrozen_source_and_changed_common_core(self):
        path = self.families['a_step'] / 'source/common.py'
        path.write_text('VALUE = 8\n')
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
            tool.prepare(self.base, self.destination, ['a_gain10'])
        self.refresh('a_step')
        with self.assertRaisesRegex(ValueError, 'Unmatched shared training core'):
            tool.prepare(self.base, self.destination, ['a_gain10'])
        self.assertFalse(self.destination.exists())

    def test_refuses_seed_filtering(self):
        root = self.families['main']; manifest = tool.read(root / 'manifest.json')
        manifest['goal']['confirmation_seeds']['cub'] = [1, 2]
        write(root / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Prespecified confirmation seeds'):
            tool.prepare(self.base, self.destination, ['a_gain10'])

    def test_does_not_clobber_existing_output(self):
        self.destination.mkdir(); marker = self.destination / 'keep'; marker.write_text('existing')
        with self.assertRaises(FileExistsError):
            tool.prepare(self.base, self.destination, ['a_gain10'])
        self.assertEqual(marker.read_text(), 'existing')

    def optional_allocation(self):
        import shutil
        reference = self.families['a_step']
        root = self.base / tool.OPTIONAL_SUITES['allocation']
        shutil.copytree(reference / 'source', root / 'source')
        (root / 'source/backbone').mkdir()
        (root / 'source/backbone/budget_allocation.py').write_text('# Optional selector fixture\n')
        (root / 'run_job.py').write_text('# Worker fixture\n')
        manifest = tool.read(reference / 'manifest.json')
        manifest['source_reference'] = str(reference)
        manifest['training_source_sha256'] = hashlib.sha256(b'allocation').hexdigest()
        jobs = []
        for job in manifest['jobs']:
            job = dict(job)
            cfg = tool.read(job['config'])
            gain = 1 if job['arm'] == 'a_gain10' else 10
            job['arm'] = f'coverage_gain{gain}'
            job['name'] = job['name'].replace('a_gain10' if gain == 1 else 'a_gain100', job['arm'])
            cfg.update(prefix=job['name'], sa_p0_run_id=job['name'],
                       sa_hoep_budget_allocation='coverage',
                       sa_a_gain_experiment=gain != 1, sa_a_gradient_multiplier=float(gain),
                       filepath=str(root/'outputs'/job['name'])+'/')
            path = root/'configs'/(job['name']+'.json'); write(path, cfg)
            job.update(config=str(path), output=cfg['filepath']); jobs.append(job)
        manifest['jobs'] = jobs
        paths = list((root/'source').rglob('*.py')) + list((root/'configs').glob('*.json')) + [root/'run_job.py']
        manifest['file_sha256'] = {str(p.relative_to(root)): tool.sha(p) for p in paths}
        write(root/'manifest.json', manifest)
        return root

    def test_optional_allocation_preserves_all_confirmation_seeds(self):
        self.optional_allocation()
        result = tool.prepare(self.base, self.destination, ['coverage_gain1', 'coverage_gain10'])
        self.assertEqual(len(result['jobs']), 12)
        self.assertEqual(sum(j['arm'] == 'frozen' for j in result['jobs']), 4)
        self.assertEqual({j['seed'] for j in result['jobs'] if j['dataset'] == 'cub'}, {2, 3})

    def test_optional_allocation_refuses_unrelated_core_change(self):
        root = self.optional_allocation()
        (root/'source/common.py').write_text('VALUE = 99\n')
        manifest = tool.read(root/'manifest.json')
        manifest['file_sha256']['source/common.py'] = tool.sha(root/'source/common.py')
        write(root/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Unmatched shared training core'):
            tool.prepare(self.base, self.destination, ['coverage_gain1'])


if __name__ == '__main__':
    unittest.main()
