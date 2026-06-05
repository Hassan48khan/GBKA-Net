"""Train GBKA-Net on a single dataset.

Example:
    python train.py --dataset TN3K --data_root data/TN3K --out_dir runs/tn3k --seed 42

Ablation variants (Tables 5-6):
    python train.py --dataset TN3K --data_root data/TN3K --out_dir runs/tn3k_nogkmsagate --no_gkmsa_gate
    python train.py --dataset TN3K --data_root data/TN3K --out_dir runs/tn3k_nogbafgate  --no_gbaf_gate
    python train.py --dataset TN3K --data_root data/TN3K --out_dir runs/tn3k_nodeepsup   --no_deepsup
    python train.py --dataset TN3K --data_root data/TN3K --out_dir runs/tn3k_mlp         --no_kan
"""
import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from datasets import UltrasoundSegDataset
from gbka_net import GBKANet
from losses import GBKADeepSupervisionLoss
from utils import set_seed, save_checkpoint, dice_score


def build_argparser():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    ap.add_argument("--batch_size", type=int, default=cfg.batch_size)
    ap.add_argument("--max_epochs", type=int, default=cfg.max_epochs)
    ap.add_argument("--early_stop_patience", type=int, default=cfg.early_stop_patience)
    ap.add_argument("--img_size", type=int, default=cfg.img_size)
    # ablation flags
    ap.add_argument("--no_gkmsa_gate", action="store_true")
    ap.add_argument("--no_gbaf_gate", action="store_true")
    ap.add_argument("--no_deepsup", action="store_true")
    ap.add_argument("--no_kan", action="store_true")
    return ap


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    dices = []
    for img, msk in loader:
        img = img.to(device)
        out = model(img)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
        gt = (msk.numpy() > 0.5).astype(np.uint8)
        for p, g in zip(pred, gt):
            dices.append(dice_score(p[0], g[0]))
    return float(np.mean(dices))


def main():
    args = build_argparser().parse_args()
    cfg = Config(seed=args.seed, lr=args.lr, weight_decay=args.weight_decay,
                 batch_size=args.batch_size, max_epochs=args.max_epochs,
                 early_stop_patience=args.early_stop_patience, img_size=args.img_size,
                 no_gkmsa_gate=args.no_gkmsa_gate, no_gbaf_gate=args.no_gbaf_gate,
                 no_deepsup=args.no_deepsup, no_kan=args.no_kan)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = UltrasoundSegDataset(args.data_root, split="train",
                                    img_size=cfg.img_size, augment=True)
    val_ds = UltrasoundSegDataset(args.data_root, split="val",
                                  img_size=cfg.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    model = GBKANet(num_classes=cfg.num_classes, input_channels=cfg.input_channels,
                    img_size=cfg.img_size, embed_dims=list(cfg.embed_dims),
                    depths=list(cfg.depths), factor=cfg.factor,
                    deep_supervision=not cfg.no_deepsup, no_kan=cfg.no_kan).to(device)

    criterion = GBKADeepSupervisionLoss(seg_gamma=cfg.seg_gamma,
                                        w_boundary=cfg.w_boundary)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    best_dice, patience = 0.0, 0
    log_path = os.path.join(args.out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_dice"])

    for epoch in range(cfg.max_epochs):
        model.train()
        running = 0.0
        for img, msk in train_loader:
            img, msk = img.to(device), msk.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss, _parts = criterion(out, msk)   # loss is scalar; _parts is a dict of components
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        train_loss = running / max(len(train_loader), 1)
        val_dice = validate(model, val_loader, device)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{val_dice:.4f}"])
        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_dice {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            patience = 0
            save_checkpoint({"model": model.state_dict(), "epoch": epoch,
                             "val_dice": best_dice}, os.path.join(args.out_dir, "best.pth"))
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best val_dice {best_dice:.4f})")
                break

    print(f"Done. Best val Dice = {best_dice:.4f}. Checkpoint: {args.out_dir}/best.pth")


if __name__ == "__main__":
    main()
