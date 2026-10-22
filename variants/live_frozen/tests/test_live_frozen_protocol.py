import json

from scripts.live_frozen_protocol import make_config, generate_configs


def test_snapshot_pair_changes_only_capture_flag_and_artifact_identity(tmp_path):
    source = {
        "dataset": "imagenetr", "seed": [1995], "batch_size": 32,
        "init_cls": 20, "increment": 20, "max_tasks": 10,
        "epochs": 20, "lora_rank": 10, "device": ["0"],
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
        "sa_use_prototype_classifier": True,
        "sa_coordinate_stable_transport": True,
        "sa_live_a_coordinate_align": True,
    }
    off = make_config(source, "inr", 1995, "live", False, tmp_path)
    on = make_config(source, "inr", 1995, "live", True, tmp_path)
    for key in ("prefix", "filepath", "sa_save_task_snapshots"):
        off.pop(key)
        on.pop(key)
    assert off == on
    assert on["batch_size"] == 128
    assert on["sa_dual_head"] is False
    assert on["sa_train_a_all_tasks"] is True


def test_generate_configs_contains_all_prespecified_paired_seeds(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    for short, dataset, init in (("c100", "cifar224", 10),
                                 ("inr", "imagenetr", 20), ("cub", "cub", 20)):
        (source_dir / f"{short}.json").write_text(json.dumps({
            "dataset": dataset, "seed": [1], "batch_size": 32,
            "init_cls": init, "increment": init,
            "epochs": 20, "lora_rank": 10,
            "sa_cumulative_merge": "live_a_aggregate_b",
            "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
            "sa_use_prototype_classifier": True,
            "sa_coordinate_stable_transport": True,
            "sa_live_a_coordinate_align": True,
        }))
    manifest = generate_configs(source_dir, tmp_path / "out")
    assert len(manifest["smoke"]) == 4
    assert len(manifest["formal"]) == 16
    pairs = {(entry["dataset"], entry["seed"], entry["policy"])
             for entry in manifest["formal"]}
    assert len(pairs) == 16
    assert ("c100", 1993, "frozen") in pairs
    assert ("inr", 1997, "live") in pairs
    assert ("cub", 3, "frozen") in pairs
    first = json.loads((tmp_path / "out" / "configs" /
                        "lfdiag_inr_live_seed1995_snapshot_on.json").read_text())
    assert first["max_tasks"] == 10
