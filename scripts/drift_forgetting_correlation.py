#!/usr/bin/env python3
"""Correlate per-task diagnostics with per-task forgetting.

Per-task forgetting is derived from the final CNN accuracy matrix: for class
group i, the drop at task t is ``max_{j<=t} M[i][j] - M[i][t]`` and the
per-task value is the mean drop over groups ``i <= t``.  Diagnostics are
parsed from the log (operator drift, gauge residual/rotation/preservation,
LRPT relative drift error).  Pearson/Spearman correlations are reported.
"""

import argparse
import glob
import math
import re


def parse_accuracy_matrix(path):
    text = open(path, errors="ignore").read()
    match = re.search(r"Accuracy Matrix \(CNN\):\s*\[\[(.*?)\]\]", text, re.S)
    if not match:
        return None
    rows = []
    for line in match.group(1).split("]"):
        values = re.findall(r"[\d.]+", line)
        if values:
            rows.append([float(v) for v in values])
    if not rows:
        return None
    width = max(len(r) for r in rows)
    matrix = []
    for row in rows:
        row = row + [float("nan")] * (width - len(row))
        matrix.append(row)
    return matrix


def per_task_forgetting(matrix):
    t = len(matrix)
    forgetting = []
    for task in range(t):
        drops = []
        for i in range(task + 1):
            best = max(matrix[i][: task + 1])
            drops.append(best - matrix[i][task])
        forgetting.append(sum(drops) / len(drops))
    return forgetting


def parse_diagnostics(path):
    text = open(path, errors="ignore").read()
    operator = {}
    residual = {}
    rotation = {}
    preservation = {}
    lrpt = {}
    for m in re.finditer(
        r"operator-stability task (\d+): relative_effective_drift=([\d.eE+-]+)",
        text,
    ):
        operator[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(
        r"cumulative gauge task (\d+): relative_projection_residual=([\d.eE+-]+) "
        r"basis_rotation_fro=([\d.eE+-]+) operator_preservation=([\d.eE+-]+)",
        text,
    ):
        task = int(m.group(1))
        residual[task] = float(m.group(2))
        rotation[task] = float(m.group(3))
        preservation[task] = float(m.group(4))
    for m in re.finditer(
        r"LRPT task (\d+): rank=\d+ reg=[\d.eE+-]+ relative_drift_error=([\d.eE+-]+)",
        text,
    ):
        lrpt[int(m.group(1))] = float(m.group(2))
    return {
        "operator": operator,
        "residual": residual,
        "rotation": rotation,
        "preservation": preservation,
        "lrpt": lrpt,
    }


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return float("nan"), float("nan")
    r = cov / math.sqrt(var_x * var_y)
    t_stat = r * math.sqrt((n - 2) / max(1 - r * r, 1e-12))
    # Two-sided p-value via normal approximation of the t distribution.
    import statistics

    from scipy.stats import t as t_dist

    try:
        p = 2 * (1 - t_dist.cdf(abs(t_stat), n - 2))
    except Exception:
        p = float("nan")
    return r, p


def spearman(xs, ys):
    try:
        from scipy.stats import spearmanr
    except Exception:
        return float("nan"), float("nan")
    return spearmanr(xs, ys)


def aligned(diag, forgetting):
    tasks = sorted(set(diag) & set(range(len(forgetting))))
    tasks = [t for t in tasks if t >= 1]
    if not tasks:
        return [], []
    return [diag[t] for t in tasks], [forgetting[t] for t in tasks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()
    for pattern in args.logs:
        for path in sorted(glob.glob(pattern)):
            matrix = parse_accuracy_matrix(path)
            if matrix is None:
                print("{:<70} no accuracy matrix".format(path))
                continue
            forgetting = per_task_forgetting(matrix)
            diag = parse_diagnostics(path)
            print("{:<70}".format(path))
            for name, values in diag.items():
                if not values:
                    continue
                xs, ys = aligned(values, forgetting)
                if len(xs) < 3:
                    print(
                        "    {:>12}: n={} points (too few)".format(name, len(xs))
                    )
                    continue
                r, p = pearson(xs, ys)
                rho, rho_p = spearman(xs, ys)
                print(
                    "    {:>12}: pearson_r={:+.3f} (p={:.3f})  spearman={:+.3f} "
                    "(p={:.3f})  n={}".format(
                        name, r, p, rho, rho_p, len(xs)
                    )
                )


if __name__ == "__main__":
    main()
