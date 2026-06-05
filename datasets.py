"""Dataset and transforms for GBKA-Net.

Expects the layout produced by prepare_data.py:
    <root>/<split>/images/<name>.png
    <root>/<split>/masks/<name>.png      (same <name>)
For test-only datasets (BUSI, UDIAT) the split folder may be omitted and the
images/masks placed directly under <root>/images and <root>/masks.
"""
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _list_pairs(root, split=None):
    base = os.path.join(root, split) if split else root
    img_dir = os.path.join(base, "images")
    msk_dir = os.path.join(base, "masks")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    pairs = []
    for ip in imgs:
        name = os.path.basename(ip)
        mp = os.path.join(msk_dir, name)
        if os.path.exists(mp):
            pairs.append((ip, mp))
    return pairs


class UltrasoundSegDataset(Dataset):
    def __init__(self, root, split=None, img_size=256, augment=False):
        self.pairs = _list_pairs(root, split)
        if len(self.pairs) == 0:
            raise RuntimeError(f"No image/mask pairs found under {root} (split={split})")
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _load(self, path, is_mask):
        flag = cv2.IMREAD_GRAYSCALE
        x = cv2.imread(path, flag)
        x = cv2.resize(x, (self.img_size, self.img_size),
                       interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR)
        return x

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = self._load(ip, is_mask=False).astype(np.float32) / 255.0
        msk = self._load(mp, is_mask=True)
        msk = (msk > 127).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy(); msk = np.fliplr(msk).copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy(); msk = np.flipud(msk).copy()

        img = torch.from_numpy(img).unsqueeze(0)            # (1, H, W)
        msk = torch.from_numpy(msk).unsqueeze(0)            # (1, H, W)
        return img, msk
