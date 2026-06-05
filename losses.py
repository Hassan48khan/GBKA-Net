"""
================================================================================
 losses.py  -  Deep-supervision loss for GBKA-Net
================================================================================

Implements BMANet's Eq. (15) total loss, adapted to GBKA-Net's five heads:

    L_total = L_seg(GT, S_g) + sum_{i in {2,3,4}} L_seg(GT, S_i) + L_b(B, S_b)

where
    L_seg = wBCE + wIoU            (boundary-weighted; emphasizes hard pixels)
    L_b   = wBCE                   (boundary-weighted BCE on the edge map)
    B     = Canny edges of the GT mask, generated on the fly (no manual labels)

The "weighted" variants follow the F3Net / BASNet convention used by BMANet:
a per-pixel weight  w = 1 + gamma * |avgpool(GT) - GT|  up-weights pixels near
boundaries and hard regions, so both BCE and IoU focus on difficult pixels.

All seg heads are supervised against the SAME full-resolution mask GT (every
head is upsampled to input size inside the model). The boundary head is
supervised against the Canny edge target.

Usage
-----
    from gbka_net import GBKANet
    from losses import GBKADeepSupervisionLoss

    model = GBKANet(num_classes=1, input_channels=1, deep_supervision=True)
    criterion = GBKADeepSupervisionLoss()

    S2, S_b, S_g, S4, S3 = model(images)          # logits, all (B,1,H,W)
    loss, parts = criterion((S2, S_b, S_g, S4, S3), masks)   # masks: (B,1,H,W) in {0,1}
    loss.backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
#  Boundary-weighted BCE + IoU  (a.k.a. structure / F3Net weighted loss)
# ----------------------------------------------------------------------------- #
def _pixel_weights(mask, gamma=5.0, ksize=31):
    """
    Per-pixel difficulty weight. Pixels whose local neighbourhood differs from
    themselves (i.e. near edges / thin structures) get larger weight.
        w = 1 + gamma * | avgpool(mask) - mask |
    mask : (B,1,H,W) float in [0,1].
    """
    pad = ksize // 2
    avg = F.avg_pool2d(mask, kernel_size=ksize, stride=1, padding=pad)
    weight = 1.0 + gamma * torch.abs(avg - mask)
    return weight


def weighted_bce(logits, target, weight=None, eps=1e-7):
    """Weighted binary cross-entropy from logits. target in {0,1}."""
    if weight is None:
        weight = torch.ones_like(target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    wbce = (weight * bce).flatten(1).sum(1) / (weight.flatten(1).sum(1) + eps)
    return wbce.mean()


def weighted_iou(logits, target, weight=None, eps=1e-7):
    """Weighted soft-IoU loss from logits. target in {0,1}."""
    if weight is None:
        weight = torch.ones_like(target)
    prob = torch.sigmoid(logits)
    inter = (prob * target * weight).flatten(1).sum(1)
    union = ((prob + target) * weight).flatten(1).sum(1) - inter
    iou = (inter + eps) / (union + eps)
    return (1.0 - iou).mean()


def seg_loss(logits, mask, gamma=5.0):
    """L_seg = wBCE + wIoU, with boundary-emphasizing pixel weights."""
    w = _pixel_weights(mask, gamma=gamma)
    return weighted_bce(logits, mask, w) + weighted_iou(logits, mask, w)


# ----------------------------------------------------------------------------- #
#  Auto-Canny boundary ground truth (differentiable-free; runs under no_grad)
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def mask_to_boundary(mask, dilation=1):
    """
    Derive a 1-pixel(+) boundary target from a binary mask WITHOUT OpenCV, using
    morphological gradient (dilate - erode) via max-pooling. Equivalent in spirit
    to Canny-on-mask for clean segmentation masks, and fully tensor-native so it
    works on GPU inside the training loop.

    mask : (B,1,H,W) float in {0,1}
    returns boundary : (B,1,H,W) float in {0,1}
    """
    k = 2 * dilation + 1
    pad = dilation
    # dilation = maxpool ; erosion = -maxpool(-x)
    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)
    boundary = (dil - ero > 0).float()
    return boundary


# ----------------------------------------------------------------------------- #
#  Full deep-supervision criterion
# ----------------------------------------------------------------------------- #
class GBKADeepSupervisionLoss(nn.Module):
    """
    Total loss for GBKA-Net.

    Args:
        seg_gamma     : weight strength for hard/boundary pixels in seg loss.
        boundary_dil  : dilation radius for the morphological boundary GT.
        w_seg_heads   : dict of weights for each seg head; defaults to 1.0 each.
        w_boundary    : weight on the boundary loss term.
    """
    def __init__(self, seg_gamma=5.0, boundary_dil=1,
                 w_seg_heads=None, w_boundary=1.0):
        super().__init__()
        self.seg_gamma = seg_gamma
        self.boundary_dil = boundary_dil
        self.w_boundary = w_boundary
        # default deep-supervision weights (final head S2 emphasized)
        self.w = w_seg_heads or {"S2": 1.0, "S3": 1.0, "S4": 1.0, "Sg": 1.0}

    def forward(self, outputs, mask):
        """
        outputs : tuple (S2, S_b, S_g, S4, S3) of logits, each (B,1,H,W).
        mask    : ground-truth segmentation (B,1,H,W) in {0,1} (float).
        """
        S2, S_b, S_g, S4, S3 = outputs
        if mask.dtype != S2.dtype:
            mask = mask.float()

        # --- segmentation heads (all vs the same mask) -----------------------
        l_s2 = seg_loss(S2, mask, self.seg_gamma)
        l_s3 = seg_loss(S3, mask, self.seg_gamma)
        l_s4 = seg_loss(S4, mask, self.seg_gamma)
        l_sg = seg_loss(S_g, mask, self.seg_gamma)

        # --- boundary head ---------------------------------------------------
        boundary_gt = mask_to_boundary(mask, dilation=self.boundary_dil)
        bw = _pixel_weights(boundary_gt, gamma=self.seg_gamma)
        l_b = weighted_bce(S_b, boundary_gt, bw)

        total = (self.w["S2"] * l_s2 + self.w["S3"] * l_s3 +
                 self.w["S4"] * l_s4 + self.w["Sg"] * l_sg +
                 self.w_boundary * l_b)

        parts = {"total": total.detach(), "S2": l_s2.detach(), "S3": l_s3.detach(),
                 "S4": l_s4.detach(), "Sg": l_sg.detach(), "Sb": l_b.detach()}
        return total, parts


# ----------------------------------------------------------------------------- #
#  Self-test
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    B, H, W = 2, 256, 256
    # fake logits and a fake circular mask
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    disk = ((xx**2 + yy**2) < 0.3).float()[None, None].repeat(B, 1, 1, 1)  # (B,1,H,W)

    outputs = tuple(torch.randn(B, 1, H, W, requires_grad=True) for _ in range(5))
    crit = GBKADeepSupervisionLoss()
    loss, parts = crit(outputs, disk)
    loss.backward()
    print("boundary GT positive frac:", mask_to_boundary(disk).mean().item())
    for k, v in parts.items():
        print(f"  {k:6s}: {v.item():.4f}")
    print("grad on S2 head present:", outputs[0].grad is not None)
