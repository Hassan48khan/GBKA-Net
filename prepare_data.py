"""Convert raw dataset downloads into the train/val/test layout used in the paper.

Usage:
    python prepare_data.py --dataset tn3k  --raw_dir <TN3K root>  --out_dir data/TN3K  --seed 42
    python prepare_data.py --dataset ddti  --raw_dir <unzipped>   --out_dir data/DDTI  --seed 42
    python prepare_data.py --dataset busi  --raw_dir <unzipped>   --out_dir data/BUSI
    python prepare_data.py --dataset udiat --raw_dir <unzipped>   --out_dir data/UDIAT

Splitting protocol (matches the paper):

  * TN3K  -- the OFFICIAL predefined partition is preserved. The 614 images in the
             official test folder form the test set and never enter training or
             model selection. The official trainval folder (2,879 images) is split
             80:20 into training (2,303) and validation (576) with a fixed seed.
             Expected raw layout (official release):
                 <raw_dir>/trainval-image/   <raw_dir>/trainval-mask/
                 <raw_dir>/test-image/       <raw_dir>/test-mask/
  * DDTI  -- no official split exists; a single fixed 70:15:15 partition
             (446 / 96 / 95) is drawn with a fixed seed.
  * BUSI / UDIAT -- written test-only (used for cross-organ evaluation).

Images are resized to 256x256 and converted to grayscale; masks are binarized at
>127. Original file names are preserved (so the test set can be checked against
the official folder) and the exact split lists are written to
<out_dir>/splits/{train,val,test}.txt.
"""
import os
import glob
import argparse
import random
import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

DDTI_SPLIT = (446, 96, 95)
TN3K_TRAINVAL_SIZE = 2879
TN3K_TEST_SIZE = 614
TN3K_VAL_FRACTION = 0.2


def _list_images(folder):
    return sorted(f for f in glob.glob(os.path.join(folder, "*"))
                  if f.lower().endswith(IMG_EXTS))


def _stem(path):
    s = os.path.splitext(os.path.basename(path))[0]
    return s.replace("_mask", "").replace("_segmentation", "")


def pair_folders(img_dir, mask_dir):
    """Pair images and masks held in two parallel folders, matched by file stem."""
    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        raise RuntimeError(f"Expected folders not found:\n  {img_dir}\n  {mask_dir}")
    masks = {_stem(m): m for m in _list_images(mask_dir)}
    return [(ip, masks[_stem(ip)]) for ip in _list_images(img_dir)
            if _stem(ip) in masks]


def pair_glob(raw_dir, img_glob, mask_glob):
    """Pair images and masks found by glob patterns (DDTI / BUSI / UDIAT)."""
    masks = sorted(glob.glob(os.path.join(raw_dir, mask_glob), recursive=True))
    mask_set = set(masks)
    imgs = [p for p in sorted(glob.glob(os.path.join(raw_dir, img_glob), recursive=True))
            if p not in mask_set]
    mask_by_stem = {_stem(m): m for m in masks}
    return [(ip, mask_by_stem[_stem(ip)]) for ip in imgs if _stem(ip) in mask_by_stem]


def write_split(pairs, out_dir, split, img_size):
    """Write one split; returns the list of source file names actually written."""
    img_out = os.path.join(out_dir, split, "images")
    msk_out = os.path.join(out_dir, split, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)
    written = []
    for ip, mp in pairs:
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            print(f"  [skip] unreadable: {ip}")
            continue
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        msk = ((msk > 127) * 255).astype(np.uint8)
        name = _stem(ip) + ".png"                 # keep the original identifier
        cv2.imwrite(os.path.join(img_out, name), img)
        cv2.imwrite(os.path.join(msk_out, name), msk)
        written.append(os.path.basename(ip))
    return written


def save_split_lists(out_dir, lists):
    split_dir = os.path.join(out_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    for split, names in lists.items():
        with open(os.path.join(split_dir, f"{split}.txt"), "w") as f:
            f.write("\n".join(sorted(names)) + "\n")
    print(f"  split lists written to {split_dir}/")


def prepare_tn3k(args):
    """Preserve the official TN3K train/test partition; split trainval 80:20."""
    trainval = pair_folders(os.path.join(args.raw_dir, "trainval-image"),
                            os.path.join(args.raw_dir, "trainval-mask"))
    test = pair_folders(os.path.join(args.raw_dir, "test-image"),
                        os.path.join(args.raw_dir, "test-mask"))

    if len(trainval) != TN3K_TRAINVAL_SIZE or len(test) != TN3K_TEST_SIZE:
        raise RuntimeError(
            f"Expected the official TN3K partition "
            f"({TN3K_TRAINVAL_SIZE} trainval / {TN3K_TEST_SIZE} test), "
            f"but found {len(trainval)} / {len(test)}. Check --raw_dir.")

    rng = random.Random(args.seed)
    trainval = sorted(trainval)
    rng.shuffle(trainval)                 # shuffle the trainval pool ONLY
    n_val = int(round(TN3K_VAL_FRACTION * len(trainval)))
    val, train = trainval[:n_val], trainval[n_val:]

    lists = {
        "train": write_split(train, args.out_dir, "train", args.img_size),
        "val":   write_split(val,   args.out_dir, "val",   args.img_size),
        "test":  write_split(test,  args.out_dir, "test",  args.img_size),  # official
    }
    save_split_lists(args.out_dir, lists)
    print(f"  train={len(lists['train'])} val={len(lists['val'])} "
          f"test={len(lists['test'])} (official test partition preserved)")


def prepare_ddti(args, pairs):
    """DDTI has no official split: one fixed 70:15:15 image-level partition."""
    n_tr, n_va, n_te = DDTI_SPLIT
    if len(pairs) < n_tr + n_va + n_te:
        raise RuntimeError(
            f"DDTI: expected at least {n_tr + n_va + n_te} pairs, found {len(pairs)}")
    rng = random.Random(args.seed)
    pairs = sorted(pairs)
    rng.shuffle(pairs)
    tr = pairs[:n_tr]
    va = pairs[n_tr:n_tr + n_va]
    te = pairs[n_tr + n_va:n_tr + n_va + n_te]
    lists = {
        "train": write_split(tr, args.out_dir, "train", args.img_size),
        "val":   write_split(va, args.out_dir, "val",   args.img_size),
        "test":  write_split(te, args.out_dir, "test",  args.img_size),
    }
    save_split_lists(args.out_dir, lists)
    print(f"  train={len(lists['train'])} val={len(lists['val'])} test={len(lists['test'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tn3k", "ddti", "busi", "udiat"])
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img_glob", default="**/*.png",
                    help="(DDTI/BUSI/UDIAT) glob relative to raw_dir matching the images")
    ap.add_argument("--mask_glob", default="**/*mask*.png",
                    help="(DDTI/BUSI/UDIAT) glob relative to raw_dir matching the masks")
    args = ap.parse_args()

    print(f"[{args.dataset}] preparing -> {args.out_dir}")

    if args.dataset == "tn3k":
        prepare_tn3k(args)
        return

    pairs = pair_glob(args.raw_dir, args.img_glob, args.mask_glob)
    if not pairs:
        raise RuntimeError("No (image, mask) pairs found. "
                           "Adjust --img_glob / --mask_glob.")
    print(f"  found {len(pairs)} pairs")

    if args.dataset == "ddti":
        prepare_ddti(args, pairs)
    else:                                  # busi / udiat: test-only
        names = write_split(sorted(pairs), args.out_dir, "test", args.img_size)
        save_split_lists(args.out_dir, {"test": names})
        print(f"  wrote {len(names)} images to {args.out_dir}/test")


if __name__ == "__main__":
    main()
