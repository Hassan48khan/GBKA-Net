"""
================================================================================
 GBKA-Net : Gated Boundary-guided KAN Attention Network
            (for thyroid nodule segmentation in ultrasound)
================================================================================

Baseline   : U-KAN  (Li et al., AAAI 2025)
Upgrade #1  : GKMSA  - Gated Kolmogorov-Arnold Multi-Scale Attention block,
                       replaces U-KAN's KANBlock.
Upgrade #2  : GBAF   - Gated Boundary-Aware Fusion, replaces the plain additive
                       skip connections with a boundary+region-gated fusion,
                       trained with deep supervision (Option A, faithful to
                       BMANet's BMA + boundary branch + side outputs).

--------------------------------------------------------------------------------
 NOVELTY MAP  (what is reused vs. what is new in this file)
--------------------------------------------------------------------------------
 Module          Borrowed from                 NEW contribution here
 -------------   ---------------------------   --------------------------------
 KANLinear       U-KAN (unchanged)             -
 GKMSABlock      EMA grouping (Ouyang 2023) +  Global->local GATE injected into
                 KanGMSA layout (Li 2026)      the 3x3 branch: the 1x1 global
                                               context map multiplies the 3x3
                                               conv output BEFORE pooling, so
                                               local detail is filtered by
                                               global anatomy. Pure-multiplicative
                                               (can fully suppress speckle).
 BoundaryBranch  BMANet BAM idea (Wu 2025):    Single-shot boundary head adapted
                 low + high feature fusion     to U-KAN's encoder taps; supervised
                                               by auto-Canny edges of the mask.
 GBAF            BMANet BMA (reverse attn +    The boundary+reverse "evidence map"
                 CBAM + attention mask)        is turned into an explicit GATE on
                                               the ENCODER skip feature, applied
                                               multiplicatively BEFORE the 3-way
                                               concat. This suppresses encoder
                                               speckle where neither boundary nor
                                               predicted-region evidence supports
                                               it -- the decoder-side analogue of
                                               the GKMSA bottleneck gate.
 Deep super.     BMANet loss (Eq. 15)          wIoU+wBCE on S_g,S4,S3,S2 and
                                               wBCE on boundary S_b. (in losses.py)
--------------------------------------------------------------------------------

 forward() returns, in BMANet order:
     S2  (final segmentation logits, full resolution),
     S_b (boundary logits),
     S_g (global-map logits),
     S4, S3 (deep-supervision side outputs)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def to_2tuple(x):
    if isinstance(x, (int, float)):
        return (int(x), int(x))
    return x


# ============================================================================= #
#  SECTION 1 - U-KAN core pieces (unchanged from baseline)
# ============================================================================= #
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class KANLinear(torch.nn.Module):
    """Edge-wise learnable B-spline activations (U-KAN, unchanged)."""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU,
                 grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h
                 + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                      - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
                     * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)])
                     * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1)
                                     if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1),
                                 self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        reg_act = l1_fake.sum()
        p = l1_fake / reg_act
        reg_ent = -torch.sum(p * p.log())
        return regularize_activation * reg_act + regularize_entropy * reg_ent


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)   # (B,N,C) -> (B,C,H,W)
        x = self.relu(self.bn(self.dwconv(x)))
        x = x.flatten(2).transpose(1, 2)         # (B,C,H,W) -> (B,N,C)
        return x


class KANLayer(nn.Module):
    """KAN -> DwConv front-end (U-KAN, unchanged). Tokens in, tokens out."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        gs, so = 5, 3
        if not no_kan:
            self.fc1 = KANLinear(in_features, hidden_features, grid_size=gs, spline_order=so)
            self.fc2 = KANLinear(hidden_features, out_features, grid_size=gs, spline_order=so)
            self.fc3 = KANLinear(hidden_features, out_features, grid_size=gs, spline_order=so)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)
        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(out_features)
        self.dwconv_3 = DW_bn_relu(out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C)).reshape(B, N, self.fc1.out_features).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, self.fc1.out_features)).reshape(B, N, self.fc2.out_features).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, self.fc2.out_features)).reshape(B, N, self.fc3.out_features).contiguous()
        x = self.dwconv_3(x, H, W)
        return x


# ============================================================================= #
#  SECTION 2 - GKMSA : Gated KAN Multi-Scale Attention (Upgrade #1)
# ============================================================================= #
class GroupedGatedMSA(nn.Module):
    """
    NOVELTY: global-to-local gating inside EMA-style grouped attention.
    The 1x1 branch's global context map (x_h.sigmoid()*x_w.sigmoid()) is used
    as a pure-multiplicative GATE on the 3x3 branch output BEFORE its pooling,
    so local detail is filtered by global anatomy and ultrasound speckle in
    the 3x3 path can be fully suppressed. (B,C,H,W) -> (B,C,H,W).
    """
    def __init__(self, channels, factor=16):
        super(GroupedGatedMSA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0, \
            f"channels ({channels}) must be >= groups ({self.groups})"
        cg = channels // self.groups
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))     # X-AvgPool -> (.,.,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))     # Y-AvgPool -> (.,.,1,W)
        self.gn = nn.GroupNorm(cg, cg)
        self.conv1x1 = nn.Conv2d(cg, cg, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(cg, cg, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        cg = c // self.groups
        group_x = x.reshape(b * self.groups, cg, h, w)          # (B*G, cg, H, W)

        # ----- 1x1 branch : global cross-channel context ----------------------
        x_h = self.pool_h(group_x)                              # (B*G, cg, H, 1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)          # (B*G, cg, W, 1)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))         # (B*G, cg, H+W, 1)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        gate_h = x_h.sigmoid()                                  # (B*G, cg, H, 1)
        gate_w = x_w.permute(0, 1, 3, 2).sigmoid()              # (B*G, cg, 1, W)
        x1 = self.gn(group_x * gate_h * gate_w)                 # (B*G, cg, H, W)

        # ----- GLOBAL-TO-LOCAL GATE (NEW) --------------------------------------
        global_gate = gate_h * gate_w                           # (B*G, cg, H, W) broadcast

        # ----- 3x3 branch : local detail, gated -------------------------------
        x2 = self.conv3x3(group_x)                              # (B*G, cg, H, W)
        x2 = x2 * global_gate                                   # >>> GATE applied <<<

        # ----- EMA cross-branch aggregation -----------------------------------
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, cg, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, cg, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        out = group_x * weights.sigmoid()
        out = out + group_x                                     # internal residual
        return out.reshape(b, c, h, w)


class GKMSABlock(nn.Module):
    """KANLayer -> inner LN -> GroupedGatedMSA, with U-KAN's outer LN + DropPath.
    Tokens (B,N,C)+(H,W) in/out. Drop-in replacement for U-KAN's KANBlock."""
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, no_kan=False, factor=16):
        super().__init__()
        self.dim = dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.kan_layer = KANLayer(in_features=dim, hidden_features=dim,
                                  out_features=dim, act_layer=act_layer,
                                  drop=drop, no_kan=no_kan)
        self.inner_norm = norm_layer(dim)
        self.attn = GroupedGatedMSA(channels=dim, factor=factor)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _block(self, x, H, W):
        B, N, C = x.shape
        x = self.kan_layer(x, H, W)                  # (B,N,C)
        x = self.inner_norm(x)                       # (B,N,C)
        x = x.transpose(1, 2).reshape(B, C, H, W)    # (B,C,H,W)
        x = self.attn(x)                             # (B,C,H,W)
        x = x.flatten(2).transpose(1, 2)             # (B,N,C)
        return x

    def forward(self, x, H, W):
        return x + self.drop_path(self._block(self.norm2(x), H, W))


# ============================================================================= #
#  SECTION 3 - Boundary branch + GBAF (Upgrade #2)
# ============================================================================= #
class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sig(self.conv(torch.cat([avg_out, max_out], dim=1)))


class _ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        red = max(channels // reduction, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(channels, red, 1, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(red, channels, 1, bias=False))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.mlp(self.avg(x)) + self.mlp(self.mx(x)))


class CBAM(nn.Module):
    """Channel-then-spatial attention (Woo 2018), used to clean fused features."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.ca = _ChannelAttention(channels, reduction)
        self.sa = _SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class BoundaryBranch(nn.Module):
    """
    NOVELTY (adapted from BMANet BAM): produce a single boundary map S_b by
    fusing a LOW-level encoder tap (rich spatial detail) with a HIGH-level
    semantic tap (good localization). Multi-kernel spatial refinement + a
    channel branch to suppress low-level noise, then a 1-channel boundary head.
    Output: boundary logits at low-level resolution (B,1,h,w).
    """
    def __init__(self, low_ch, high_ch, inter_ch=64):
        super().__init__()
        self.reduce_low = nn.Conv2d(low_ch, inter_ch, 1)
        self.reduce_high = nn.Conv2d(high_ch, inter_ch, 1)
        self.fuse = nn.Conv2d(inter_ch * 2, inter_ch, 1)
        # spatial multi-scale branch
        self.c3 = nn.Conv2d(inter_ch, inter_ch, 3, padding=1)
        self.c5 = nn.Conv2d(inter_ch, inter_ch, 5, padding=2)
        self.c7 = nn.Conv2d(inter_ch, inter_ch, 7, padding=3)
        self.sa = _SpatialAttention()
        # channel branch (noise suppression on low-level features)
        self.ca = _ChannelAttention(inter_ch)
        self.head = nn.Conv2d(inter_ch, 1, 1)

    def forward(self, low_feat, high_feat):
        # bring high-level feature up to low-level spatial size
        high_up = F.interpolate(high_feat, size=low_feat.shape[2:],
                                mode='bilinear', align_corners=False)
        lo = self.reduce_low(low_feat)                       # (B, inter, h, w)
        hi = self.reduce_high(high_up)                       # (B, inter, h, w)
        f = self.fuse(torch.cat([lo, hi], dim=1))            # (B, inter, h, w)
        fs = self.c3(f) + self.c5(f) + self.c7(f)            # multi-scale spatial
        fs = fs * self.sa(fs)                                # spatial-attended
        fc = f * self.ca(f)                                  # channel-attended
        sb = self.head(fs + fc)                              # (B, 1, h, w) boundary logits
        return sb


class GBAF(nn.Module):
    """
    Gated Boundary-Aware Fusion  (NOVELTY - the core decoder contribution).

    Inputs:
      enc_feat : encoder skip feature  f'_i        (B, C, H, W)
      dec_pred : prediction from adjacent higher level S_{i+1}  (B, 1, h', w')
      bound    : boundary map S_b                  (B, 1, h_b, w_b)

    Faithful-to-BMANet pieces:
      * reverse attention   : S^r = 1 - sigmoid(S_{i+1})
      * three-way evidence  : region (pred), reverse (1-pred), boundary
      * CBAM cleanup + 1x1 prediction head, residual on encoder feature

    >>> THE GATE (new) <<<
      The boundary + region + reverse signals are combined into a single
      spatial EVIDENCE GATE g in (0,1). g multiplies the ENCODER feature
      BEFORE the three-way concat (pure multiplicative -> can fully suppress).
      Rationale: the coarse encoder skip carries ultrasound speckle; we keep it
      only where boundary or predicted-region evidence supports it. This is the
      decoder-side analogue of the GKMSA bottleneck gate.

    Output:
      side_pred : segmentation logits S_i at encoder resolution (B, 1, H, W)
      fused     : refined feature (B, C, H, W)   (passed on as next decoder feat)
    """
    def __init__(self, channels):
        super().__init__()
        self.pred_proj = nn.Conv2d(1, 1, 1)             # learnable scale on region map
        # evidence gate: maps the 3 evidence channels -> 1 spatial gate
        self.gate_conv = nn.Sequential(
            nn.Conv2d(3, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        # three-way fusion of {gated-bg, gated-region, gated-boundary} features
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.attn_mask = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.cbam = CBAM(channels)
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, enc_feat, dec_pred, bound):
        residual = enc_feat
        size = enc_feat.shape[2:]

        # bring prediction & boundary to encoder resolution
        pred = F.interpolate(dec_pred, size=size, mode='bilinear', align_corners=False)
        pred = torch.sigmoid(pred)                          # (B,1,H,W) region prob
        rev = 1.0 - pred                                    # reverse attention
        edge = torch.sigmoid(
            F.interpolate(bound, size=size, mode='bilinear', align_corners=False))  # (B,1,H,W)

        # ---------------- EVIDENCE GATE (NEW) ----------------------------------
        # Combine the three spatial evidences into one gate in (0,1).
        evidence = torch.cat([pred, rev, edge], dim=1)      # (B, 3, H, W)
        gate = self.gate_conv(evidence)                     # (B, 1, H, W)
        # Gate the encoder feature (pure multiplicative; can suppress speckle).
        enc_gated = enc_feat * gate                         # (B, C, H, W)  >>> GATE <<<

        # ---------------- three-way boundary-guided fusion ---------------------
        region_feat = enc_gated * self.pred_proj(pred)      # region-emphasized
        bg_feat = enc_gated * rev                           # reverse/background
        edge_feat = enc_gated * edge                        # boundary-emphasized
        fused = self.fusion(torch.cat([bg_feat, region_feat, edge_feat], dim=1))

        # attention mask + residual + CBAM cleanup (BMANet-style)
        fused = fused * self.attn_mask(fused)
        fused = fused + residual
        fused = self.cbam(fused)

        side_pred = self.head(fused)                        # (B, 1, H, W) logits
        return side_pred, fused


# ============================================================================= #
#  SECTION 4 - PatchEmbed & conv layers (U-KAN, unchanged)
# ============================================================================= #
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size); patch_size = to_2tuple(patch_size)
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


# ============================================================================= #
#  SECTION 5 - GBKA-Net
# ============================================================================= #
class GBKANet(nn.Module):
    """
    Gated Boundary-guided KAN Attention Network.

    Encoder : 3 conv stages (t1,t2,t3) + 2 tokenized GKMSA stages (t4, bottleneck)
    Boundary: BoundaryBranch fuses t1 (low) with the bottleneck (high) -> S_b
    Decoder : GBAF replaces the three deepest additive skips (t4, t3, t2);
              t1 skip + final upsample kept as in U-KAN.
    Deep sup: returns (S2, S_b, S_g, S4, S3).
    """
    def __init__(self, num_classes=1, input_channels=1, deep_supervision=True,
                 img_size=256, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], factor=16, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        c0 = embed_dims[0]

        # ----- encoder conv stages -------------------------------------------
        self.encoder1 = ConvLayer(input_channels, c0 // 8)   # -> t1  (c0/8)
        self.encoder2 = ConvLayer(c0 // 8, c0 // 4)          # -> t2  (c0/4)
        self.encoder3 = ConvLayer(c0 // 4, c0)               # -> t3  (c0)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ----- tokenized GKMSA stages ----------------------------------------
        self.block1 = nn.ModuleList([GKMSABlock(dim=embed_dims[1], drop=drop_rate,
                       drop_path=dpr[i], norm_layer=norm_layer, no_kan=no_kan,
                       factor=factor) for i in range(depths[0])])
        self.block2 = nn.ModuleList([GKMSABlock(dim=embed_dims[2], drop=drop_rate,
                       drop_path=dpr[sum(depths[:1]) + i], norm_layer=norm_layer,
                       no_kan=no_kan, factor=factor) for i in range(depths[1])])
        self.dblock1 = nn.ModuleList([GKMSABlock(dim=embed_dims[1], drop=drop_rate,
                       drop_path=dpr[sum(depths[:2]) + i], norm_layer=norm_layer,
                       no_kan=no_kan, factor=factor) for i in range(depths[2])])
        self.dblock2 = nn.ModuleList([GKMSABlock(dim=embed_dims[0], drop=drop_rate,
                       drop_path=dpr[i], norm_layer=norm_layer, no_kan=no_kan,
                       factor=factor) for i in range(depths[0])])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                       in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                       in_chans=embed_dims[1], embed_dim=embed_dims[2])

        # ----- decoder conv stages -------------------------------------------
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])   # 512 -> 320
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])   # 320 -> 256
        self.decoder3 = D_ConvLayer(embed_dims[0], c0 // 4)         # 256 -> 64
        self.decoder4 = D_ConvLayer(c0 // 4, c0 // 8)               # 64  -> 32
        self.decoder5 = D_ConvLayer(c0 // 8, c0 // 8)               # 32  -> 32

        # ----- global map head (S_g) from bottleneck -------------------------
        self.global_head = nn.Conv2d(embed_dims[2], 1, 1)

        # ----- boundary branch : low=t1 (c0/8), high=bottleneck (c2) ---------
        self.boundary = BoundaryBranch(low_ch=c0 // 8, high_ch=embed_dims[2], inter_ch=64)

        # ----- GBAF at the three deepest skips -------------------------------
        self.gbaf4 = GBAF(channels=embed_dims[1])   # fuses t4 (320)
        self.gbaf3 = GBAF(channels=embed_dims[0])   # fuses t3 (256)
        self.gbaf2 = GBAF(channels=c0 // 4)         # fuses t2 (64)

        self.final = nn.Conv2d(c0 // 8, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        in_size = x.shape[2:]

        # ===== Encoder =======================================================
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2)); t1 = out      # (B, c0/8, H/2,  W/2)
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2)); t2 = out    # (B, c0/4, H/4,  W/4)
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2)); t3 = out    # (B, c0,   H/8,  W/8)

        # ===== Tokenized GKMSA stage 1 =======================================
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()   # (B, 320, H/16, W/16)
        t4 = out

        # ===== Tokenized GKMSA stage 2 (bottleneck) ==========================
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        bottleneck = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B, 512, H/32, W/32)

        # ===== Global map S_g & Boundary S_b =================================
        S_g = self.global_head(bottleneck)                                # (B,1,H/32,W/32)
        S_b = self.boundary(t1, bottleneck)                               # (B,1,H/2, W/2)

        # ===== Decoder with GBAF skips =======================================
        # decoder1: 512 -> 320, upsample to t4 size, GBAF fuse with t4
        d = F.relu(F.interpolate(self.decoder1(bottleneck), scale_factor=2, mode='bilinear'))  # (B,320,.,.)
        S4, d = self.gbaf4(t4, S_g, S_b)        # GBAF uses encoder t4; d-feat refined
        # tokenized decoder block on the GBAF-refined feature
        _, _, H, W = d.shape
        d_tok = d.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            d_tok = blk(d_tok, H, W)
        d = self.dnorm3(d_tok).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B,320,.,.)

        # decoder2: 320 -> 256, upsample, GBAF fuse with t3 (use S4 as adjacent pred)
        d = F.relu(F.interpolate(self.decoder2(d), scale_factor=2, mode='bilinear'))  # (B,256,.,.)
        S3, d = self.gbaf3(t3, S4, S_b)
        _, _, H, W = d.shape
        d_tok = d.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            d_tok = blk(d_tok, H, W)
        d = self.dnorm4(d_tok).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B,256,.,.)

        # decoder3: 256 -> 64, upsample, GBAF fuse with t2 (use S3 as adjacent pred)
        d = F.relu(F.interpolate(self.decoder3(d), scale_factor=2, mode='bilinear'))  # (B,64,.,.)
        S2_low, d = self.gbaf2(t2, S3, S_b)

        # decoder4/5: 64 -> 32 -> 32, plain t1 skip + final upsample (U-KAN tail)
        d = F.relu(F.interpolate(self.decoder4(d), scale_factor=2, mode='bilinear'))  # (B,32,.,.)
        d = torch.add(d, t1)
        d = F.relu(F.interpolate(self.decoder5(d), scale_factor=2, mode='bilinear'))  # (B,32,H,W)
        S2 = self.final(d)                                                            # (B,1,H,W)

        # ===== Resize all heads to input resolution for supervision ==========
        S_g = F.interpolate(S_g, size=in_size, mode='bilinear', align_corners=False)
        S_b = F.interpolate(S_b, size=in_size, mode='bilinear', align_corners=False)
        S4  = F.interpolate(S4,  size=in_size, mode='bilinear', align_corners=False)
        S3  = F.interpolate(S3,  size=in_size, mode='bilinear', align_corners=False)

        if self.deep_supervision:
            return S2, S_b, S_g, S4, S3
        return S2

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        total = 0.0
        for m in self.modules():
            if isinstance(m, KANLinear):
                total += m.regularization_loss(regularize_activation, regularize_entropy)
        return total


# ============================================================================= #
#  Smoke test
# ============================================================================= #
if __name__ == "__main__":
    model = GBKANet(num_classes=1, input_channels=1, img_size=256,
                    embed_dims=[256, 320, 512], depths=[1, 1, 1], factor=16,
                    deep_supervision=True)
    x = torch.randn(2, 1, 256, 256)
    outs = model(x)
    names = ["S2 (final)", "S_b (boundary)", "S_g (global)", "S4", "S3"]
    for n, o in zip(names, outs):
        print(f"{n:16s}: {tuple(o.shape)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")
