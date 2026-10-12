"""Restart provenance and unstable GPU exclusion."""
import importlib
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def implementation():
    return importlib.import_module('scripts.hoep_cifar_stable_queue')


def test_all_lanes_exclude_unstable_cards():
    assert implementation().PAIRS == ((0, 1), (2, 5), (6, 7))
    assert set(g for pair in implementation().PAIRS for g in pair) == {0, 1, 2, 5, 6, 7}


def test_second_model_requires_memory_reservation_and_capacity():
    f=implementation().pair_can_dispatch
    pair=(0,1); readings={0:(9000,100),1:(8900,90)}
    active={'job':{'gpu_pair':pair}}
    assert not f(pair,active,readings,max_jobs=1)
    assert f(pair,active,readings,max_jobs=2,reserved_mb=11000,budget_mb=23000)
    assert not f(pair,active,{0:(13000,100),1:(8900,90)},max_jobs=2,reserved_mb=11000,budget_mb=23000)
    assert not f(pair,{'a':{'gpu_pair':pair},'b':{'gpu_pair':pair}},readings,max_jobs=2)


def test_empty_pair_must_be_idle_even_if_memory_budget_would_fit():
    f=implementation().pair_can_dispatch
    assert f((0,1),{}, {0:(1,0),1:(1,0)},max_jobs=2)
    assert not f((0,1),{}, {0:(9000,0),1:(1,0)},max_jobs=2)
    assert not f((0,1),{}, {0:(1,60),1:(1,0)},max_jobs=2)


def test_relocating_config_preserves_all_scientific_settings(tmp_path):
    old = {'prefix': 'old', 'filepath': '/old/', 'seed': [1994],
           'batch_size': 64, 'epochs': 20, 'device': ['0', '1'],
           'sa_hoep_reference_counts_path': '/old/counts.json', 'sa_resume': False}
    new = implementation().relocate_config(old, 'new', tmp_path / 'output', tmp_path / 'counts.json')
    assert new['prefix'] == 'new'
    assert new['filepath'] == str(tmp_path / 'output') + '/'
    assert new['sa_hoep_reference_counts_path'] == str(tmp_path / 'counts.json')
    for key in ('seed', 'batch_size', 'epochs', 'device', 'sa_resume'):
        assert new[key] == old[key]
    assert old['prefix'] == 'old'


def test_reboot_archives_interrupted_files_without_overwriting(tmp_path):
    log, output, proof = tmp_path / 'run.log', tmp_path / 'output', tmp_path / 'proof.json'
    output.mkdir(); (output / 'state.pt').write_bytes(b'partial checkpoint')
    log.write_text('partial log'); proof.write_text('{"phase":"running"}')
    entry = {'name': 'run', 'log': str(log), 'output': str(output)}
    archived = implementation().archive_interrupted(entry, proof, tmp_path / 'attempts')
    assert (Path(archived) / 'run.log').read_text() == 'partial log'
    assert (Path(archived) / 'output/state.pt').read_bytes() == b'partial checkpoint'
    assert json.loads((Path(archived) / 'proof.json').read_text())['phase'] == 'running'
    assert not log.exists() and not output.exists() and not proof.exists()


def test_config_verifier_rejects_changed_training_setting(tmp_path):
    old = {'prefix': 'old', 'filepath': '/old/', 'epochs': 20, 'batch_size': 64}
    new = {'prefix': 'new', 'filepath': '/new/', 'epochs': 10, 'batch_size': 64}
    with pytest.raises(ValueError, match='scientific'):
        implementation().verify_relocation(old, new)


def test_worker_cannot_start_training_before_ownership_is_saved(tmp_path):
    module = implementation()
    proof, marker = tmp_path / 'proof.json', tmp_path / 'marker'
    source = "import os,sys,json,pathlib; fd=int(sys.argv[-1]); token=os.read(fd,1); " \
             "p=pathlib.Path(sys.argv[1]); " \
             "sys.exit(0) if not token else None; " \
             "assert json.loads(p.read_text())['phase']=='starting'; " \
             "pathlib.Path(sys.argv[2]).write_text('started')"
    command = [sys.executable, '-c', source, str(proof), str(marker)]
    process = module.launch_owned_worker(command, 'run', (2, 5), proof)
    assert process.wait(timeout=3) == 0
    assert marker.read_text() == 'started'
    assert json.loads(proof.read_text())['worker_pid'] == process.pid


def test_failed_ownership_write_never_releases_training_gate(tmp_path, monkeypatch):
    module = implementation()
    proof, marker = tmp_path / 'proof.json', tmp_path / 'marker'
    source = "import os,sys,pathlib; token=os.read(int(sys.argv[-1]),1); " \
             "pathlib.Path(sys.argv[1]).write_text('started') if token else None"

    def fail(*args):
        raise OSError('disk failure')

    monkeypatch.setattr(module, 'atomic_json', fail)
    with pytest.raises(OSError, match='disk failure'):
        module.launch_owned_worker([sys.executable, '-c', source, str(marker)], 'run', (0, 1), proof)
    assert not marker.exists()
