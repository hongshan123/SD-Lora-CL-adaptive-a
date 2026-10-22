"""Recovery must require completed subprocess events, logs and artifacts."""
import importlib
import json
from pathlib import Path

import pytest


def implementation():
    return importlib.import_module('scripts.resume_live_frozen_cuda6')


def fixture_run(tmp_path):
    entry = {'config': str(tmp_path / 'run.json'), 'snapshots': True}
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs/run.log').write_text(
        'CNN top1 curve: [90, 89, 88, 87, 86, 85, 84, 83, 82, 81]\nForgetting (CNN): 3\n')
    snap = tmp_path / 'results/run/task_snapshots/task_009'
    snap.mkdir(parents=True)
    (snap / 'complete.json').write_text('{}')
    event = {'kind': 'formal_training_complete', 'name': 'run'}
    return entry, [event]


def test_completed_artifact_without_zero_exit_event_is_not_reused(tmp_path):
    entry, events = fixture_run(tmp_path)
    module = implementation()
    assert not module.training_complete(tmp_path, entry, [])
    assert module.training_complete(tmp_path, entry, events)


def test_partial_curve_or_missing_last_snapshot_is_not_reused(tmp_path):
    entry, events = fixture_run(tmp_path)
    module = implementation()
    (tmp_path / 'results/run/task_snapshots/task_009/complete.json').unlink()
    assert not module.training_complete(tmp_path, entry, events)
    (tmp_path / 'results/run/task_snapshots/task_009/complete.json').write_text('{}')
    (tmp_path / 'logs/run.log').write_text('CNN top1 curve: [90, 89]\nForgetting (CNN): 3\n')
    assert not module.training_complete(tmp_path, entry, events)


def test_diagnosis_requires_matching_event_and_saved_logits(tmp_path):
    entry, _ = fixture_run(tmp_path)
    folder = tmp_path / 'diagnostics/run'
    folder.mkdir(parents=True)
    (folder / 'task_001.pt').write_bytes(b'logits')
    (folder / 'task_001.json').write_text(json.dumps({'task': 1, 'logits_file': str(folder / 'task_001.pt')}))
    events = [{'kind': 'diagnosis_complete', 'name': 'run', 'task': 1}]
    module = implementation()
    assert module.diagnosis_complete(tmp_path, entry, 1, events)
    assert not module.diagnosis_complete(tmp_path, entry, 2, events)
    assert not module.diagnosis_complete(tmp_path, entry, 1, [])
    (folder / 'task_001.pt').unlink()
    assert not module.diagnosis_complete(tmp_path, entry, 1, events)


def test_archive_interrupted_preserves_complete_sibling(tmp_path):
    entry, _ = fixture_run(tmp_path)
    sibling = tmp_path / 'results/other'
    sibling.mkdir()
    (sibling / 'state.pt').write_bytes(b'complete')
    archive = implementation().archive_interrupted(tmp_path, entry)
    assert (archive / 'run.log').is_file()
    assert (archive / 'output/task_snapshots/task_009/complete.json').is_file()
    assert not (tmp_path / 'results/run').exists()
    assert (sibling / 'state.pt').read_bytes() == b'complete'
