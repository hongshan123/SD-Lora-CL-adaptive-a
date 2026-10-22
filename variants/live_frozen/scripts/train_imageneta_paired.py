#!/usr/bin/env python3
"""Run the unchanged trainer after checking each loaded pretrained backbone."""

import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_base_weights(model, expected):
    from utils.canonical_hash import hash_named_tensors

    actual = hash_named_tensors(dict(model.state_dict()))
    if actual != expected:
        raise ValueError(f"pretrained weights differ from sealed base: {actual} != {expected}")
    return actual


def main():
    import timm

    expected = os.environ["INA_BASE_WEIGHT_SHA256"]
    original = timm.create_model

    def checked_create_model(name, *args, **kwargs):
        model = original(name, *args, **kwargs)
        if name == "vit_base_patch16_224" and kwargs.get("pretrained"):
            actual = check_base_weights(model, expected)
            print(f"[SealedBaseHash] rank={os.environ.get('RANK', '0')} sha256={actual} PASS", flush=True)
        return model

    timm.create_model = checked_create_model
    try:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="__main__")
    finally:
        timm.create_model = original


if __name__ == "__main__":
    main()
