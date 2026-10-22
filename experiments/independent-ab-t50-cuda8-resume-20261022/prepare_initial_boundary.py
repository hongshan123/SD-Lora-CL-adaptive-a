"""Recover the last fully evaluated boundary without changing old artifacts."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import torch

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/home/hongzhijun/hongshan/independent-ab-t50-cuda8-retry-20261021/independent_bank')


def decode(text):
    return ast.literal_eval(re.sub(r'np\.(?:float|int)(?:32|64)\(([-+0-9.eE]+)\)', r'\1', text))


def main():
    target = ROOT / 'initial_boundary'
    assert not target.exists()
    target.mkdir()
    output = ORIGINAL / 'output'
    log = (ORIGINAL / 'formal.log').read_text(errors='replace')
    curve = {}
    for kind in ['top1', 'top5']:
        curve[kind] = decode(re.findall(r'CNN ' + kind + r' curve:\s*(\[[^\n]+\])', log)[-1])
    assert len(curve['top1']) == len(curve['top5']) == 42
    groups = [decode(text) for text in re.findall(r'CNN: (\{[^\n]+\})', log)]
    matrix = [[float(record[k]) for k in sorted(k for k in record if '-' in k)] for record in groups]
    assert len(matrix) == 42 and all(len(row) == i+1 for i, row in enumerate(matrix))
    order = decode(re.findall(r'\[data_manager.py\].*?=> (\[[^\n]+\])', log)[0])
    assert len(order) == len(set(order)) == 100
    state = torch.load(output / 'sa_state.pt', map_location='cpu', weights_only=True)
    assert state['structure_mode'] == 'independent_bank' and len(state['entries']) == 43
    state['entries'] = state['entries'][:42]
    state['task_id'] = 41
    state['shared_a'] = state['entries'][-1]['down']
    for name in ['sa_state.pt', 'sa_merged_lora.pt']:
        torch.save(state, target / name)
    prototypes = torch.load(output / 'sa_prototypes.pt', map_location='cpu', weights_only=True)
    assert set(prototypes) == set(range(84))
    shutil.copy2(output / 'sa_prototypes.pt', target / 'sa_prototypes.pt')
    shutil.copy2(output / 'CLs_weight41.pt', target / 'fc_weight.pt')
    shutil.copy2(output / 'CLs_bias41.pt', target / 'fc_bias.pt')
    hashes = json.loads((output / 'p0_hashes.json').read_text())
    replay = {k: v for k, v in hashes['42'].items() if k.startswith('post_train_')}
    hashes = {str(i): hashes[str(i)] for i in range(42)}
    assert all('post_eval_model_hash' in value for value in hashes.values())
    (target / 'p0_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    diagnostics = json.loads((output / 'sa_coordinate_diagnostics.json').read_text())
    assert len(diagnostics) == 41 and diagnostics[-1]['task_id'] == 41
    (target / 'sa_coordinate_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    metadata = dict(version=1, mode='independent_bank', tasks_completed=42, seed=1993,
                    class_order=order, model_hash=hashes['41']['post_eval_model_hash'],
                    cnn_curve=curve, nme_curve={'top1': [], 'top5': []},
                    cnn_matrix=matrix, nme_matrix=[], original_runtime=str(ORIGINAL),
                    original_log_sha256=hashlib.sha256((ORIGINAL/'formal.log').read_bytes()).hexdigest(),
                    reconstruction='First 42 immutable independent A/B entries, shared_a from entry41, FC41 and untouched 84-class prototypes')
    (target / 'boundary.json').write_text(json.dumps(metadata, indent=2)+'\n')
    config = json.loads((ROOT / 'base_config.json').read_text())
    config['sa_resume_replay_expected'] = {'42': replay}
    (ROOT / 'base_config.json').write_text(json.dumps(config, indent=2)+'\n')
    print('RECOVERED_INITIAL_BOUNDARY',42,'tasks',84,'classes','last_top1',curve['top1'][-1],flush=True)


if __name__ == '__main__':
    main()
