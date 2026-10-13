"""A5 architecture components, preserved from the manuscript implementation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------- ASPP ---
class ASPP(nn.Module):
    """Parallel dilated convolutions plus an image-level branch.

    The rates 6/12/18 only make sense if the feature map is large enough to hold a
    3x3 kernel dilated by 18, i.e. 37 px. The encoder is therefore run at output
    stride 16 (layer4 dilated, not strided), giving 32x32 for a 512 input. At stride
    32 the map would be 16x16 and the widest branch would degenerate into an
    expensive 1x1.
    """

    def __init__(self, cin, cout=256, rates=(6, 12, 18), dropout=0.1):
        super().__init__()

        def branch(d):
            if d == 1:
                conv = nn.Conv2d(cin, cout, 1, bias=False)
            else:
                conv = nn.Conv2d(cin, cout, 3, padding=d, dilation=d, bias=False)
            return nn.Sequential(conv, nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.branches = nn.ModuleList([branch(1)] + [branch(d) for d in rates])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(cout * (len(rates) + 2), cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Dropout2d(dropout))

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        g = self.pool(x)
        feats.append(F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, 1))


# ---------------------------------------------------------------- decoder ---
def conv_bn_relu(cin, cout, k=3):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    """Upsample, concatenate the skip if there is one, then two 3x3 convolutions."""

    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv1 = conv_bn_relu(cin + cskip, cout)
        self.conv2 = conv_bn_relu(cout, cout)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], 1)
        return self.conv2(self.conv1(x))


class ResidualBlock(nn.Module):
    """Stride-1 residual block. The edge branch is built from a stack of these."""

    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.short = (nn.Identity() if cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False),
                                    nn.BatchNorm2d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.short(x))


# ----------------------------------------------------------------- fusion ---
class GatedFusion(nn.Module):
    """Per-pixel choice between the context feature and the detail feature.

    g = sigmoid(Conv1x1([Fs ; Fe])),  out = g * Fs' + (1 - g) * Fe'

    Both streams are projected to `cout` first so the gate mixes comparable things.
    A wall interior should lean on the semantic branch and a handrail on the edge
    branch, and the gate is what lets one model do both.
    """

    def __init__(self, c_sem, c_edge, cout):
        super().__init__()
        self.p_sem = conv_bn_relu(c_sem, cout, k=1)
        self.p_edge = conv_bn_relu(c_edge, cout, k=1)
        self.gate = nn.Conv2d(cout * 2, cout, 1)
        self.out = conv_bn_relu(cout, cout)

    def forward(self, f_sem, f_edge):
        s, e = self.p_sem(f_sem), self.p_edge(f_edge)
        g = torch.sigmoid(self.gate(torch.cat([s, e], 1)))
        return self.out(g * s + (1.0 - g) * e)


# --------------------------------------------------- region-consistent layer ---
class RegionConsistentLayer(nn.Module):
    """Soft, learned version of 'one flood-filled cell, one label'.

    Inputs
        probs   (B, K, H, W) softmax over classes
        rid     (B, H, W) int64, values in [0, n_regions). Slot 0 is ink / no cell
                and always passes through untouched.

    Returns a dict with
        probs       (B, K, H, W) the corrected distribution
        cell_mean   (B, R, K)    mean probability per cell, for the cell-level loss
        cell_count  (B, R)       pixels per cell, 0 for unused slots
        alpha       (B, R)       how much of each cell was handed to its mean

    tau and temp are learned. temp is stored through softplus so it stays positive;
    a temp near zero reproduces the hard 0.80 vote exactly.
    """

    def __init__(self, n_classes, tau=0.80, temp=0.05):
        super().__init__()
        self.n_classes = n_classes
        self.tau = nn.Parameter(torch.tensor(float(tau)))
        # softplus(raw) = temp  ->  raw = log(exp(temp) - 1)
        self._temp_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(temp)))))

    @property
    def temp(self):
        return F.softplus(self._temp_raw) + 1e-4

    def forward(self, probs, rid, n_regions=None):
        B, K, H, W = probs.shape
        R = int(n_regions if n_regions is not None else int(rid.max().item()) + 1)
        flat = rid.reshape(B, -1)                                   # (B, N)
        p = probs.permute(0, 2, 3, 1).reshape(B, -1, K)             # (B, N, K)

        idx = flat.unsqueeze(-1).expand(-1, -1, K)
        sums = torch.zeros(B, R, K, dtype=p.dtype, device=p.device)
        sums.scatter_add_(1, idx, p)
        cnt = torch.zeros(B, R, dtype=p.dtype, device=p.device)
        cnt.scatter_add_(1, flat, torch.ones_like(flat, dtype=p.dtype))
        mean = sums / cnt.clamp(min=1.0).unsqueeze(-1)

        conf = mean.max(dim=-1).values                              # (B, R)
        alpha = torch.sigmoid((conf - self.tau) / self.temp)
        alpha = alpha * (cnt > 0)                                   # unused slots
        alpha = alpha.clone()
        alpha[:, 0] = 0.0                                           # ink passes through

        a_pix = alpha.gather(1, flat).unsqueeze(-1)                 # (B, N, 1)
        m_pix = mean.gather(1, idx)                                 # (B, N, K)
        out = a_pix * m_pix + (1.0 - a_pix) * p
        out = out.reshape(B, H, W, K).permute(0, 3, 1, 2).contiguous()
        return dict(probs=out, cell_mean=mean, cell_count=cnt, alpha=alpha)


def region_consistent_numpy(probs, rid, n_regions, tau, temp):
    """The same arithmetic in numpy, for full-image inference.

    Inference assembles the whole drawing's probabilities on the CPU (a 20 MP image
    at 17 classes is 700 MB in fp16), so the layer has to run there too. Keeping a
    second implementation is only safe because tests/test_region_layer.py asserts the
    two agree; do not edit one without the other.

    probs (K, H, W) float32, rid (H, W) int32/int64. Returns (K, H, W) float32.

    Written class by class and in place: a whole drawing is up to 11.8 MP here, and
    materialising (K, N) in float64 would cost 1.6 GB for no accuracy that survives
    the argmax.
    """
    K = probs.shape[0]
    flat = np.ascontiguousarray(rid).reshape(-1).astype(np.int64, copy=False)
    cnt = np.bincount(flat, minlength=n_regions).astype(np.float64)
    mean = np.empty((n_regions, K), np.float64)
    for k in range(K):
        mean[:, k] = np.bincount(flat, weights=probs[k].reshape(-1).astype(np.float64),
                                 minlength=n_regions)
    mean /= np.maximum(cnt, 1.0)[:, None]

    conf = mean.max(axis=1)
    alpha = 1.0 / (1.0 + np.exp(-(conf - tau) / temp))
    alpha[cnt == 0] = 0.0
    alpha[0] = 0.0

    a = alpha[flat].astype(np.float32)
    out = np.empty_like(probs, dtype=np.float32)
    meanf = mean.astype(np.float32)
    for k in range(K):
        pk = probs[k].reshape(-1).astype(np.float32)
        pk *= (1.0 - a)
        pk += a * meanf[flat, k]
        out[k] = pk.reshape(probs.shape[1:])
    return out
