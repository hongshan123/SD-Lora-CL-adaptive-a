"""DomainNet manifest loading and class-incremental protocol helpers."""

from pathlib import Path

import numpy as np


DEFAULT_DOMAINNET_DOMAINS = (
    "clipart",
    "infograph",
    "painting",
    "quickdraw",
    "real",
    "sketch",
)


def select_top_classes(targets, top_k=200):
    """Return the most frequent class ids, with class-id tie breaking."""
    targets = np.asarray(targets)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional array")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")

    classes, counts = np.unique(targets.astype(np.int64), return_counts=True)
    order = np.lexsort((classes, -counts))
    return classes[order[: min(int(top_k), len(classes))]].astype(int).tolist()


def balanced_task_increments(num_classes, num_tasks):
    """Partition classes into exactly ``num_tasks`` balanced task sizes."""
    num_classes = int(num_classes)
    num_tasks = int(num_tasks)
    if num_classes <= 0 or num_tasks <= 0 or num_tasks > num_classes:
        raise ValueError(
            "num_classes and num_tasks must be positive with num_tasks <= num_classes"
        )
    base, remainder = divmod(num_classes, num_tasks)
    return [base + 1] * remainder + [base] * (num_tasks - remainder)


def _read_manifest(root, split, domains):
    paths = []
    targets = []
    lists_dir = Path(root) / "lists"
    for domain in domains:
        manifest = lists_dir / "{}_{}.txt".format(domain, split)
        if not manifest.is_file():
            raise FileNotFoundError("missing DomainNet manifest: {}".format(manifest))
        for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    "invalid DomainNet manifest line {}:{}".format(
                        manifest, line_number
                    )
                )
            relative_path, label = fields
            paths.append(str(Path(root) / relative_path))
            targets.append(int(label))
    return np.asarray(paths, dtype=object), np.asarray(targets, dtype=np.int64)


def prepare_domainnet_data(root, top_k=200, domains=None):
    """Load DomainNet, keep top train-frequency classes, and compact labels.

    The class ranking is computed from the union of the selected training
    domains.  Both splits are filtered by the same raw class ids and then
    relabeled to ``0..K-1`` in descending train-frequency order.
    """
    root = Path(root)
    domains = tuple(domains or DEFAULT_DOMAINNET_DOMAINS)
    train_paths, train_raw_targets = _read_manifest(root, "train", domains)
    test_paths, test_raw_targets = _read_manifest(root, "test", domains)
    selected = select_top_classes(train_raw_targets, top_k=top_k)
    mapping = {raw_id: new_id for new_id, raw_id in enumerate(selected)}

    def filter_and_relabel(paths, targets):
        keep = np.isin(targets, np.asarray(selected, dtype=np.int64))
        kept_paths = paths[keep]
        kept_targets = np.asarray(
            [mapping[int(label)] for label in targets[keep]], dtype=np.int64
        )
        return kept_paths, kept_targets

    train_paths, train_targets = filter_and_relabel(
        train_paths, train_raw_targets
    )
    test_paths, test_targets = filter_and_relabel(test_paths, test_raw_targets)
    if len(train_targets) == 0 or len(test_targets) == 0:
        raise ValueError("DomainNet top-class selection produced an empty split")
    return (
        train_paths,
        train_targets,
        test_paths,
        test_targets,
        selected,
    )
