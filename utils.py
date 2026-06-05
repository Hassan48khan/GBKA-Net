"""Metrics, seeding, and checkpoint utilities for GBKA-Net."""
import os
import random
import numpy as np
import torch
from medpy.metric.binary import hd95 as _hd95


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model, path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model


# ----------------------------------------------------------------------------- #
#  Segmentation metrics. pred/gt are binary numpy arrays (H, W) in {0, 1}.
# ----------------------------------------------------------------------------- #
def _confusion(pred, gt):
    tp = float(np.logical_and(pred == 1, gt == 1).sum())
    tn = float(np.logical_and(pred == 0, gt == 0).sum())
    fp = float(np.logical_and(pred == 1, gt == 0).sum())
    fn = float(np.logical_and(pred == 0, gt == 1).sum())
    return tp, tn, fp, fn


def dice_score(pred, gt, eps=1e-6):
    tp, _, fp, fn = _confusion(pred, gt)
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def iou_score(pred, gt, eps=1e-6):
    tp, _, fp, fn = _confusion(pred, gt)
    return (tp + eps) / (tp + fp + fn + eps)


def precision_score(pred, gt, eps=1e-6):
    tp, _, fp, _ = _confusion(pred, gt)
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred, gt, eps=1e-6):
    tp, _, _, fn = _confusion(pred, gt)
    return (tp + eps) / (tp + fn + eps)


def accuracy_score(pred, gt, eps=1e-6):
    tp, tn, fp, fn = _confusion(pred, gt)
    return (tp + tn + eps) / (tp + tn + fp + fn + eps)


def hd95_score(pred, gt, spacing=1.0):
    """95th-percentile Hausdorff distance (mm given pixel spacing).
    Returns NaN if either mask is empty (undefined)."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    return float(_hd95(pred.astype(bool), gt.astype(bool), voxelspacing=spacing))


def all_metrics(pred, gt, spacing=1.0):
    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "hd95": hd95_score(pred, gt, spacing),
        "precision": precision_score(pred, gt),
        "recall": recall_score(pred, gt),
        "accuracy": accuracy_score(pred, gt),
    }
