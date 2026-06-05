"""Convert raw Kaggle downloads into the train/val/test layout used in the paper.

Usage:
    python prepare_data.py --dataset tn3k  --raw_dir <unzipped> --out_dir data/TN3K  --seed 42
    python prepare_data.py --dataset ddti  --raw_dir <unzipped> --out_dir data/DDTI  --seed 42
    python prepare_data.py --dataset busi  --raw_dir <unzipped> --out_dir data/BUSI  --seed 42
    python prepare_data.py --dataset udiat --raw_dir <unzipped> --out_dir data/UDIAT --seed 42

The script resizes everything to 256x256, converts images to grayscale, binarizes
masks, and writes the fixed splits (DDTI 446/96/95, TN3K 2303/576/614). BUSI and
UDIAT are written test-only.

NOTE: Kaggle archives differ in their internal folder names. The --img_glob and
--mask_glob flags let you point the script at the correct subfolders if the
auto-detection below does not match your download.
"""
import os
import glob
import argparse
import random
import cv2
import numpy as np

SPLITS = {
    "tn3k": (2303, 576, 614),
    "ddti": (446, 96, 95),
    "busi": None,      # test-only
    "udiat": None,     # test-only
}


def find_pairs(raw_dir, img_glob, mask_glob):
    """Return list of (image_path, mask_path) by matching basenames."""
    imgs = sorted(glob.glob(os.path.join(raw_dir, img_glob), recursive=True))
    masks = sorted(glob.glob(os.path.join(raw_dir, mask_glob), recursive=True))
    # match by stem (filename without extension / without _mask suffix)
    def stem(p):
        s = os.path.splitext(os.path.basename(p))[0]
        return s.replace("_mask", "").replace("_segmentation", "")
    mask_by_stem = {stem(m): m for m in masks}
    pairs = []
    for ip in imgs:
        s = stem(ip)
        if s in mask_by_stem:
            pairs.append((ip, mask_by_stem[s]))
    return pairs


def write_split(pairs, out_dir, split, img_size):
    img_out = os.path.join(out_dir, split, "images")
    msk_out = os.path.join(out_dir, split, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)
    for i, (ip, mp) in enumerate(pairs):
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        msk = ((msk > 127) * 255).astype(np.uint8)
        name = f"{i:05d}.png"
        cv2.imwrite(os.path.join(img_out, name), img)
        cv2.imwrite(os.path.join(msk_out, name), msk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SPLITS.keys()))
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img_glob", default="**/*.png",
                    help="glob (relative to raw_dir) matching the images")
    ap.add_argument("--mask_glob", default="**/*mask*.png",
                    help="glob (relative to raw_dir) matching the masks")
    args = ap.parse_args()

    random.seed(args.seed)
    pairs = find_pairs(args.raw_dir, args.img_glob, args.mask_glob)
    if len(pairs) == 0:
        raise RuntimeError(
            "No (image, mask) pairs found. Adjust --img_glob / --mask_glob to "
            "match this dataset's folder structure.")
    random.shuffle(pairs)
    print(f"[{args.dataset}] found {len(pairs)} pairs")

    split_sizes = SPLITS[args.dataset]
    if split_sizes is None:
        # test-only dataset: write everything under <out>/test
        write_split(pairs, args.out_dir, "test", args.img_size)
        print(f"  wrote {len(pairs)} images to {args.out_dir}/test")
    else:
        n_tr, n_va, n_te = split_sizes
        tr = pairs[:n_tr]
        va = pairs[n_tr:n_tr + n_va]
        te = pairs[n_tr + n_va:n_tr + n_va + n_te]
        write_split(tr, args.out_dir, "train", args.img_size)
        write_split(va, args.out_dir, "val", args.img_size)
        write_split(te, args.out_dir, "test", args.img_size)
        print(f"  train={len(tr)} val={len(va)} test={len(te)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
