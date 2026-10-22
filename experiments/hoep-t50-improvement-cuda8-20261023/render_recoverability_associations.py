"""Render saved CPU diagnostic results; no Torch or scientific recomputation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(root):
    root = Path(root)
    data = root / 'records/long_sequence_recoverability_audit.json'
    result = json.loads(data.read_text())
    fields = [
        ('history', 'weighted_frobenius_projection_residual_ratio', 'History: own-input weighted residual'),
        ('history', 'frobenius_projection_residual_ratio', 'History: Frobenius residual'),
        ('bmetric_current', 'frobenius_projection_residual_ratio', 'B-current: Frobenius residual'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, layout='constrained')
    for axis, (method, field, title) in zip(axes, fields):
        rows = [r for r in result['rows'] if r['method'] == method]
        scatter = axis.scatter([max(r[field], 1e-30) for r in rows],
                               [r['excess_old_drop_pp'] for r in rows],
                               c=[r['task_id'] for r in rows], cmap='viridis',
                               vmin=1, vmax=49, s=25, alpha=.8)
        axis.set_xscale('log')
        axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.), numticks=7))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.0e}'))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(axis='x', labelsize=9)
        axis.axhline(0, color='black', linewidth=.8)
        axis.axhspan(-.02, .02, color='gray', alpha=.15)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel('Normalized squared projection residual')
        axis.grid(alpha=.2)
        stat = next(s for s in result['associations'] if s['method'] == method
                    and s['predictor'] == field and s['outcome'] == 'excess_old_drop_pp')
        coefficient = 'NA' if stat['spearman'] is None else f"{stat['spearman']:.3f}"
        axis.text(.02, .98, f"n={stat['n']}; Spearman={coefficient}",
                  transform=axis.transAxes, va='top', fontsize=9)
    axes[0].set_ylabel('Extra old-class accuracy drop vs Frozen (pp)')
    fig.colorbar(scatter, ax=axes, label='Task index (dependent observations)', shrink=.8)
    fig.suptitle('CUB seed1: exploratory post-deployment associations\n'
                 'Gray band: +/-0.02 pp rounding range, not a confidence interval', fontsize=11)
    base = root / 'records/long_sequence_recoverability_scatter'
    files = [base.with_suffix('.png'), base.with_suffix('.svg')]
    fig.savefig(files[0], dpi=180)
    fig.savefig(files[1])
    plt.close(fig)
    record = {
        'updated': time.time(), 'python': sys.executable,
        'matplotlib_version': matplotlib.__version__,
        'scientific_result_sha256': sha(data),
        'scientific_source_sha256': result['script_sha256'],
        'renderer_source_sha256': sha(Path(__file__)),
        'recomputed_scientific_results': False,
        'figures': {p.name: sha(p) for p in files},
    }
    (root / 'records/long_sequence_recoverability_render.json').write_text(
        json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(render(parser.parse_args().root), indent=2))
