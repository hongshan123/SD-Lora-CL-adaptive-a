#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def safe_ext(path):
    ext = Path(path or '').suffix.lower()
    if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
        return ext
    return '.jpg'


def hardlink_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=1993)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--verify-images', action='store_true')
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    output_dir = Path(args.output_dir)
    all_dir = output_dir / 'all'
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'

    files = sorted(parquet_dir.glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f'No parquet files found in {parquet_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)
    all_dir.mkdir(parents=True, exist_ok=True)

    by_class = defaultdict(list)
    total = 0
    for part_idx, parquet_file in enumerate(files):
        table = pq.read_table(parquet_file, columns=['image', 'wnid'])
        rows = table.to_pylist()
        print(f'Exporting {parquet_file.name}: {len(rows)} rows')
        for row_idx, row in enumerate(rows):
            wnid = row['wnid']
            image = row['image'] or {}
            image_bytes = image.get('bytes')
            image_path = image.get('path')
            if not wnid or image_bytes is None:
                raise ValueError(f'Bad row in {parquet_file}: {row_idx}')
            cls_dir = all_dir / wnid
            cls_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha1(image_bytes).hexdigest()[:12]
            out_name = f'{part_idx:02d}_{row_idx:05d}_{digest}{safe_ext(image_path)}'
            out_path = cls_dir / out_name
            if not out_path.exists():
                out_path.write_bytes(image_bytes)
            by_class[wnid].append(out_path)
            total += 1

    if args.verify_images:
        print('Verifying exported images with Pillow...')
        for cls, paths in sorted(by_class.items()):
            for path in paths:
                with Image.open(path) as im:
                    im.verify()

    for split_dir in [train_dir, test_dir]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    summary = {
        'source': str(parquet_dir),
        'output': str(output_dir),
        'seed': args.seed,
        'test_ratio': args.test_ratio,
        'num_classes': len(by_class),
        'total_images': total,
        'classes': {},
    }

    for cls, paths in sorted(by_class.items()):
        paths = sorted(paths)
        rng.shuffle(paths)
        n_test = max(1, int(round(len(paths) * args.test_ratio)))
        test_paths = sorted(paths[:n_test])
        train_paths = sorted(paths[n_test:])
        for src in train_paths:
            hardlink_or_copy(src, train_dir / cls / src.name)
        for src in test_paths:
            hardlink_or_copy(src, test_dir / cls / src.name)
        summary['classes'][cls] = {
            'total': len(paths),
            'train': len(train_paths),
            'test': len(test_paths),
        }

    summary_path = output_dir / 'split_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    print(f'Done: {total} images, {len(by_class)} classes')
    print(f'Train dir: {train_dir}')
    print(f'Test dir: {test_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
