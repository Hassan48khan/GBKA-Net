"""Cross-organ generalization: train on one dataset, test on another with no
fine-tuning. Reproduces Section 4.4 (thyroid->breast) and the reversed shift.

Examples:
    # thyroid -> breast (model already trained on TN3K)
    python cross_domain.py --train TN3K --test BUSI  --ckpt runs/tn3k/best.pth --test_root data/BUSI
    python cross_domain.py --train TN3K --test UDIAT --ckpt runs/tn3k/best.pth --test_root data/UDIAT

    # reversed shift (model already trained on BUSI)
    python cross_domain.py --train BUSI --test TN3K --ckpt runs/busi/best.pth --test_root data/TN3K
    python cross_domain.py --train BUSI --test DDTI --ckpt runs/busi/best.pth --test_root data/DDTI

This script only evaluates; train the source model first with train.py.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from datasets import UltrasoundSegDataset
from gbka_net import GBKANet
from utils import set_seed, load_checkpoint, all_metrics


@torch.no_grad()
def run(model, loader, device):
    model.eval()
    keys = ["dice", "iou", "hd95", "precision", "recall", "accuracy"]
    acc = {k: [] for k in keys}
    for img, msk in loader:
        img = img.to(device)
        out = model(img)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
        gt = (msk.numpy() > 0.5).astype(np.uint8)
        for p, g in zip(pred, gt):
            m = all_metrics(p[0], g[0])
            for k in keys:
                if not np.isnan(m[k]):
                    acc[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="source dataset name (for logging)")
    ap.add_argument("--test", required=True, help="target dataset name (for logging)")
    ap.add_argument("--ckpt", required=True, help="checkpoint trained on the source")
    ap.add_argument("--test_root", required=True, help="prepared target dataset root")
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()

    cfg = Config(img_size=args.img_size)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # target datasets may be test-only (no split subfolder)
    try:
        ds = UltrasoundSegDataset(args.test_root, split="test", img_size=cfg.img_size)
    except RuntimeError:
        ds = UltrasoundSegDataset(args.test_root, split=None, img_size=cfg.img_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    model = GBKANet(num_classes=cfg.num_classes, input_channels=cfg.input_channels,
                    img_size=cfg.img_size, embed_dims=list(cfg.embed_dims),
                    depths=list(cfg.depths), factor=cfg.factor,
                    deep_supervision=cfg.deep_supervision).to(device)
    load_checkpoint(model, args.ckpt, device)

    res = run(model, loader, device)
    print(f"\nCross-domain  train={args.train}  ->  test={args.test}  "
          f"({len(ds)} images):")
    print(f"  Dice = {res['dice']*100:.2f} | IoU = {res['iou']*100:.2f} | "
          f"HD95 = {res['hd95']:.2f}px")


if __name__ == "__main__":
    main()
