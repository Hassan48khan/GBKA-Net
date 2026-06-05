"""Evaluate a trained GBKA-Net checkpoint on a test set.

Reports Dice, IoU, HD95, Precision, Recall, Accuracy (mean +/- std over the
test set). Optionally measures parameters, FLOPs, and inference time.

Example:
    python evaluate.py --dataset TN3K --data_root data/TN3K --ckpt runs/tn3k/best.pth
    python evaluate.py --dataset TN3K --data_root data/TN3K --ckpt runs/tn3k/best.pth --measure_efficiency
"""
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from datasets import UltrasoundSegDataset
from gbka_net import GBKANet
from utils import set_seed, load_checkpoint, all_metrics


@torch.no_grad()
def evaluate(model, loader, device):
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
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


@torch.no_grad()
def measure_efficiency(model, device, img_size=256, n_warmup=10, n_runs=100):
    model.eval()
    x = torch.randn(1, 1, img_size, img_size, device=device)
    for _ in range(n_warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / n_runs * 1000.0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    flops = None
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x).total() / 1e9
    except Exception:
        pass
    return n_params, flops, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--measure_efficiency", action="store_true")
    args = ap.parse_args()

    cfg = Config(img_size=args.img_size)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = UltrasoundSegDataset(args.data_root, split="test",
                                   img_size=cfg.img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=cfg.num_workers)

    model = GBKANet(num_classes=cfg.num_classes, input_channels=cfg.input_channels,
                    img_size=cfg.img_size, embed_dims=list(cfg.embed_dims),
                    depths=list(cfg.depths), factor=cfg.factor,
                    deep_supervision=cfg.deep_supervision).to(device)
    load_checkpoint(model, args.ckpt, device)

    res = evaluate(model, test_loader, device)
    print(f"\nResults on {args.dataset} test set ({len(test_ds)} images):")
    print(f"  Dice      : {res['dice'][0]*100:.2f} +/- {res['dice'][1]*100:.2f}")
    print(f"  IoU       : {res['iou'][0]*100:.2f} +/- {res['iou'][1]*100:.2f}")
    print(f"  HD95 (px) : {res['hd95'][0]:.2f} +/- {res['hd95'][1]:.2f}")
    print(f"  Precision : {res['precision'][0]*100:.2f} +/- {res['precision'][1]*100:.2f}")
    print(f"  Recall    : {res['recall'][0]*100:.2f} +/- {res['recall'][1]*100:.2f}")
    print(f"  Accuracy  : {res['accuracy'][0]*100:.2f} +/- {res['accuracy'][1]*100:.2f}")

    if args.measure_efficiency:
        n_params, flops, ms = measure_efficiency(model, device, cfg.img_size)
        print(f"\nEfficiency:")
        print(f"  Params    : {n_params:.2f} M")
        print(f"  FLOPs     : {flops:.2f} G" if flops else "  FLOPs     : (install fvcore)")
        print(f"  Inference : {ms:.2f} ms/image")


if __name__ == "__main__":
    main()
