import json

from scripts.audit_live_frozen_smoke import audit_pair, parse_curve


def test_parse_curve_accepts_numpy_scalar_spelling():
    text = "CNN top1 curve: [82.17, 76.62, np.float64(74.84)]\n"
    assert parse_curve(text) == [82.17, 76.62, 74.84]


def test_audit_pair_detects_snapshot_trajectory_change(tmp_path):
    off, on = tmp_path / "off", tmp_path / "on"
    off.mkdir()
    on.mkdir()
    records = {str(task): {"post_train_model_hash": f"m{task}",
                           "post_train_rng_hash": f"r{task}",
                           "post_eval_model_hash": f"e{task}",
                           "post_eval_rng_hash": f"q{task}"}
               for task in range(10)}
    (off / "p0_hashes.json").write_text(json.dumps(records))
    (on / "p0_hashes.json").write_text(json.dumps(records))
    log = "CNN top1 curve: [" + ", ".join(["80.00"] * 9 +
          ["np.float64(79.99)"]) + "]\nForgetting (CNN): 2.0\n"
    assert audit_pair(off, on, log, log)["passed"]
    records["1"]["post_train_rng_hash"] = "changed"
    (on / "p0_hashes.json").write_text(json.dumps(records))
    result = audit_pair(off, on, log, log)
    assert result["passed"] is False
    assert result["first_mismatch"] == "task 1 post_train_rng_hash"
