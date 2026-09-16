import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.domainnet import (
    balanced_task_increments,
    prepare_domainnet_data,
    select_top_classes,
)
from utils.toolkit import accuracy


def test_select_top_classes_uses_train_frequency_then_class_id():
    train_targets = np.array([3, 3, 3, 1, 1, 2, 2, 2, 2, 0])

    selected = select_top_classes(train_targets, top_k=3)

    assert selected == [2, 3, 1]


def test_balanced_task_increments_exactly_partition_classes():
    assert balanced_task_increments(200, 5) == [40, 40, 40, 40, 40]
    assert balanced_task_increments(200, 10) == [20] * 10
    assert balanced_task_increments(200, 20) == [10] * 20
    assert balanced_task_increments(200, 40) == [5] * 40


def test_balanced_task_increments_distributes_remainder_front_loaded():
    increments = balanced_task_increments(100, 40)

    assert len(increments) == 40
    assert sum(increments) == 100
    assert increments[:20] == [3] * 20
    assert increments[20:] == [2] * 20


def test_prepare_domainnet_data_filters_and_compacts_labels(tmp_path):
    root = tmp_path / "DomainNet"
    lists = root / "lists"
    lists.mkdir(parents=True)
    (root / "real" / "cat").mkdir(parents=True)
    (root / "real" / "dog").mkdir(parents=True)
    (root / "real" / "bird").mkdir(parents=True)
    (root / "real" / "cat" / "cat.jpg").touch()
    (root / "real" / "dog" / "dog.jpg").touch()
    (root / "real" / "bird" / "bird.jpg").touch()
    for split in ("train", "test"):
        (lists / f"real_{split}.txt").write_text(
            "real/cat/cat.jpg 7\n"
            "real/dog/dog.jpg 3\n"
            "real/bird/bird.jpg 11\n"
        )

    train_paths, train_targets, test_paths, test_targets, selected = (
        prepare_domainnet_data(root, top_k=2, domains=("real",))
    )

    assert selected == [3, 7]
    assert set(train_paths.tolist()) == {
        str(root / "real/dog/dog.jpg"),
        str(root / "real/cat/cat.jpg"),
    }
    assert sorted(set(train_targets.tolist())) == [0, 1]
    assert sorted(set(test_targets.tolist())) == [0, 1]
    assert len(test_paths) == 2


def test_accuracy_omits_unobserved_future_task_groups():
    grouped = accuracy(
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
        nb_old=0,
        increment=3,
        task_increments=[3, 2, 2],
    )

    assert list(grouped) == ["total", "00-02", "old", "new"]


def test_tasklen_generator_creates_result_parent(tmp_path):
    from scripts.generate_tasklen_pair_configs import main

    runtime_root = tmp_path / "runtime"
    main(["generate_tasklen_pair_configs.py", str(Path(__file__).resolve().parents[1]), str(runtime_root)])

    assert (runtime_root / "results").is_dir()
