#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_bbox_classes.py

Utility to inspect how many unique class IDs appear in each *_bbox.npy file
inside the eight-class dataset structure:

data/eightclass/
 ├── train/
 │    ├── train_day_XXXX_bbox.npy
 │    └── ...
 ├── val/
 └── test/

Each .npy file is assumed to be a 2-D array where the **last column** encodes
`class_id`.  Adjust `cls_col_idx` below if your layout differs.

Run from project root:
    python -m utils.inspect_bbox_classes \\
        /home/oeoiewt/ebc/eTraM/rvt_eTram/data/eightclass
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def inspect_file(
    path: Path, cls_col_idx: int = -1, field_name: str = "class_id"
) -> Counter:
    """
    Return Counter of class IDs inside a single *_bbox.npy file.

    Handles two layouts:
    1. 2D numeric array where class_id is a column (default last col).
    2. 1D structured array / recarray with 'class_id' field.
    """
    arr = np.load(path, allow_pickle=True)
    # Case 1: 2‑D numeric
    if arr.ndim == 2:
        cls_ids = arr[:, cls_col_idx].astype(int)
    # Case 2: 1‑D structured array with named fields
    elif arr.ndim == 1 and arr.dtype.names and field_name in arr.dtype.names:
        cls_ids = arr[field_name].astype(int)
    else:
        raise ValueError(
            f"{path} shape={arr.shape}, dtype={arr.dtype} not recognised. "
            "Expected 2-D numeric or 1-D struct with 'class_id'."
        )
    return Counter(cls_ids.tolist())


def inspect_folder(root: Path, cls_col_idx: int = -1) -> None:
    """
    Print class histograms for every split (train/val/test) and grand total.

    Parameters
    ----------
    root : Path
        Path to `data/eightclass`
    cls_col_idx : int
        Column index where class_id is stored (default last column)
    """
    splits = ["train", "val", "test"]
    grand_total = Counter()

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"[warn] split '{split}' not found under {root}")
            continue

        split_counter = Counter()
        for npy in split_dir.glob("*_bbox.npy"):
            split_counter += inspect_file(npy, cls_col_idx, "class_id")

        grand_total += split_counter

        print(f"\n=== {split.upper()} ===")
        if split_counter:
            for cls, cnt in sorted(split_counter.items()):
                print(f"class {cls}: {cnt}")
            print(f"total samples: {sum(split_counter.values())}")
        else:
            print("No *_bbox.npy files found.")

    print("\n=== OVERALL ===")
    for cls, cnt in sorted(grand_total.items()):
        print(f"class {cls}: {cnt}")
    print(f"Grand total samples: {sum(grand_total.values())}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect class IDs in *_bbox.npy files."
    )
    parser.add_argument(
        "eightclass_dir", type=Path, help="Path to data/eightclass directory."
    )
    parser.add_argument(
        "--cls-col-idx",
        type=int,
        default=-1,
        help="Column index that contains class_id (default: last column)",
    )
    args = parser.parse_args()

    inspect_folder(args.eightclass_dir, args.cls_col_idx)


if __name__ == "__main__":
    main()


# import numpy as np

# npz_path = (
#     "/home/oeoiewt/eTraM/rvt_eTram/data/gen_processed/train/"
#     "train_day_0023/labels_v2/labels.npz"
# )

# data = np.load(npz_path, allow_pickle=True)
# class_ids = data["labels"]["class_id"].astype(int)  # → 1-D array
# uniq, counts = np.unique(class_ids, return_counts=True)

# print(f"Unique classes ({len(uniq)}): {uniq.tolist()}")
# for u, c in zip(uniq, counts):
#     print(f"  class {u}: {c}")


# import numpy as np

# npz_path = (
#     "/home/oeoiewt/eTraM/rvt_eTram/data/gen_processed/train/"
#     "train_day_0023/labels_v2/labels.npz"
# )

# data = np.load(npz_path, allow_pickle=True)
# class_ids = data["labels"]["class_id"].astype(int)  # → 1-D array
# uniq, counts = np.unique(class_ids, return_counts=True)

# print(f"Unique classes ({len(uniq)}): {uniq.tolist()}")
# for u, c in zip(uniq, counts):
#     print(f"  class {u}: {c}")
