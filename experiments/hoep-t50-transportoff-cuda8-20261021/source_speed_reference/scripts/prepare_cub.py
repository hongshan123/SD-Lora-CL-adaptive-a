#!/usr/bin/env python3
"""Prepare CUB-200-2011 into data/cub/train and data/cub/test ImageFolder layout.

Uses the official train_test_split.txt (1=train, 0=test).  Images are
symbolically linked, so the original download is preserved.
"""

import argparse
import os
import shutil
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, help="path to CUB_200_2011.tgz")
    parser.add_argument(
        "--out-root",
        default="data/cub",
        help="output root containing train/ and test/",
    )
    parser.add_argument(
        "--workdir",
        default="data/_downloads/cub_extract",
        help="temporary extraction directory",
    )
    args = parser.parse_args()

    archive = Path(args.archive)
    out_root = Path(args.out_root)
    workdir = Path(args.workdir)
    if not archive.exists():
        raise FileNotFoundError(archive)

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    print("extracting {} ...".format(archive))
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(workdir)
    root = workdir / "CUB_200_2011"
    if not root.exists():
        raise RuntimeError("unexpected CUB layout under {}".format(workdir))

    images = {}
    for line in (root / "images.txt").read_text().strip().splitlines():
        image_id, rel = line.split()
        images[int(image_id)] = rel
    labels = {}
    for line in (root / "image_class_labels.txt").read_text().strip().splitlines():
        image_id, label = line.split()
        labels[int(image_id)] = int(label)
    split = {}
    for line in (root / "train_test_split.txt").read_text().strip().splitlines():
        image_id, is_train = line.split()
        split[int(image_id)] = int(is_train)

    if set(images) != set(labels) != set(split):
        raise RuntimeError("CUB metadata id sets do not match")

    counts = {"train": 0, "test": 0}
    for image_id in sorted(images):
        rel = images[image_id]
        label = labels[image_id]
        is_train = split[image_id]
        class_name = rel.split("/")[0]
        subdir = "train" if is_train else "test"
        target_dir = out_root / subdir / "{:03d}_{}".format(label, class_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        src = root / "images" / rel
        dst = target_dir / Path(rel).name
        if not dst.exists():
            os.symlink(os.path.abspath(src), dst)
        counts[subdir] += 1

    print("train images: {}".format(counts["train"]))
    print("test images : {}".format(counts["test"]))
    print("class dirs  : {} (train) / {} (test)".format(
        len(list((out_root / "train").iterdir())),
        len(list((out_root / "test").iterdir())),
    ))


if __name__ == "__main__":
    main()
