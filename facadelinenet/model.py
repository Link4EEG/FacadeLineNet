"""A5 architecture components, preserved from the manuscript implementation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CROP, Stage, TrainCfg, ctx_geometry
from .modules import (ASPP, DecoderBlock, GatedFusion, RegionConsistentLayer,
                      ResidualBlock, conv_bn_relu)


# ------------------------------------------------------------------ input ---
def _adapt_stem(conv1, in_channels):
    """Widen a pretrained 3-channel stem to `in_channels` without losing the weights.

    Extra channels are initialised with the mean of the RGB filters, which is the
    standard trick and starts the new channel as 'a grayscale copy of what the
    pretrained filter already looks for' rather than as noise.
    """
    if in_channels == 3:
        return conv1
    w = conv1.weight.data
    new = nn.Conv2d(in_channels, conv1.out_channels, conv1.kernel_size,
                    conv1.stride, conv1.padding, bias=False)
    with torch.no_grad():
        new.weight[:, :3] = w
        for c in range(3, in_channels):
            new.weight[:, c] = w.mean(1)
    return new


def _dilate(layer, dilation=2):
    """Turn a strided ResNet stage into a dilated one, in place.

    torchvision refuses `replace_stride_with_dilation` on ResNet-18/34 because its
    BasicBlock raises on dilation > 1. The restriction is in the constructor, not in
    the maths - a 3x3 convolution dilates fine - so the stage is rewired afterwards,
    following torchvision's own Bottleneck convention: the first block loses its
    stride but keeps dilation 1, and every later block takes the new dilation.

    Without this, layer4 runs at output stride 32 and a 512 input leaves a 16x16
    bottleneck, where an ASPP branch dilated by 18 samples nothing but zero padding.
    """
    for i, block in enumerate(layer):
        if i == 0:
            block.conv1.stride = (1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
            continue
        for conv in (block.conv1, block.conv2):
            if conv.kernel_size == (3, 3):
                conv.dilation = (dilation, dilation)
                conv.padding = (dilation, dilation)
    return layer


# -------------------------------------------------------- semantic branch ---
class SemanticBranch(nn.Module):
    """Pretrained ResNet-34 at output stride 16, ASPP bottleneck, U-Net decoder.

    Output stride 16 (layer4 dilated instead of strided) is what keeps the ASPP rates
    6/12/18 - the DeepLabv3+ defaults for this stride - meaningful. A 512 input then
    leaves a 32x32 bottleneck; at stride 32 it would be 16x16 and the widest branch
    would read nothing but zero padding.
    """

    def __init__(self, in_channels, cfg: TrainCfg):
        super().__init__()
        import torchvision

        weights = None
        if cfg.encoder_weights not in ("none", None, ""):
            from torchvision.models import ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet34(weights=weights)

        self.stem = nn.Sequential(_adapt_stem(net.conv1, in_channels), net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2      # stride 4 / 8
        self.layer3 = net.layer3                               # stride 16
        self.layer4 = _dilate(net.layer4)                      # stride 16, dilated

        d = cfg.decoder_channels
        self.aspp = ASPP(512, cfg.aspp_channels, cfg.aspp_rates, cfg.dropout)
        self.d16 = nn.Sequential(conv_bn_relu(cfg.aspp_channels + 256, d[0]),
                                 conv_bn_relu(d[0], d[0]))
        self.d8 = DecoderBlock(d[0], 128, d[1])
        self.d4 = DecoderBlock(d[1], 64, d[2])
        self.d2 = DecoderBlock(d[2], 64, d[3])
        self.d1 = DecoderBlock(d[3], 0, d[4])
        self.out_channels = (d[2], d[3], d[4])                 # stride 4 / 2 / 1

    def forward(self, x):
        s2 = self.stem(x)                       # stride 2,  64
        e1 = self.layer1(self.pool(s2))         # stride 4,  64
        e2 = self.layer2(e1)                    # stride 8,  128
        e3 = self.layer3(e2)                    # stride 16, 256
        e4 = self.layer4(e3)                    # stride 16, 512 (dilated)
        b = self.aspp(e4)
        d = self.d16(torch.cat([b, e3], 1))
        d = self.d8(d, e2)
        f4 = self.d4(d, e1)
        f2 = self.d2(f4, s2)
        f1 = self.d1(f2)
        return f4, f2, f1


# ------------------------------------------------------------ edge branch ---
class EdgeBranch(nn.Module):
    """Shallow, high-resolution, exactly one downsample.

    Two downsamples would turn the 2 px stroke this branch exists to protect into
    0.5 px, which is the failure the whole native-resolution decision avoids.
    """

    def __init__(self, in_channels, channels=(32, 64, 128)):
        super().__init__()
        c0, c1, c2 = channels
        self.head = nn.Sequential(conv_bn_relu(in_channels, c0), conv_bn_relu(c0, c0))
        self.res1 = nn.Sequential(ResidualBlock(c0, c1), ResidualBlock(c1, c1))
        self.down = nn.Sequential(nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(c1), nn.ReLU(inplace=True))
        self.res2 = nn.Sequential(ResidualBlock(c1, c2), ResidualBlock(c2, c2))
        self.out_channels = (c1, c2)                            # stride 1 / 2

    def forward(self, x):
        h = self.res1(self.head(x))             # stride 1
        return h, self.res2(self.down(h))       # stride 1, stride 2


# ============================================================ FacadeLineNet ===
class FacadeLineNet(nn.Module):
    """The whole figure. `stage` decides which blocks are live."""

    def __init__(self, n_classes: int, stage: Stage, cfg: TrainCfg = TrainCfg()):
        super().__init__()
        self.n_classes = n_classes
        self.stage = stage
        self.cfg = cfg
        _, self.ctx_off, self.ctx_size = ctx_geometry(CROP, stage.ctx_scale)

        self.semantic = SemanticBranch(stage.in_channels, cfg)
        c4, c2, c1 = self.semantic.out_channels

        if stage.edge_branch:
            self.edge = EdgeBranch(stage.in_channels, cfg.edge_channels)
            e1, e2 = self.edge.out_channels
            self.fuse2 = GatedFusion(c2, e2, 64)
            self.pre1 = conv_bn_relu(c1 + 64, 32)
            self.fuse1 = GatedFusion(32, e1, 32)
        else:
            self.edge = None
            self.lift2 = conv_bn_relu(c2, 64)
            self.pre1 = conv_bn_relu(c1 + 64, 32)

        self.head = nn.Conv2d(32, n_classes, 1)
        self.aux2 = nn.Conv2d(64, n_classes, 1)     # deep supervision, stride 2
        self.aux4 = nn.Conv2d(c4, n_classes, 1)     # deep supervision, stride 4
        self.region = RegionConsistentLayer(n_classes) if stage.region_layer else None

    # ------------------------------------------------------------ helpers ---
    def _to_detail(self, feat, stride):
        """Crop a context-space feature back to the detail crop, then upsample.

        With ctx_scale=1 this is the identity. With ctx_scale=4 the detail crop is
        the middle 128x128 of the 512x512 context image, so at stride s the slice is
        [192/s : 320/s] - integral for s in 1,2,4,8,16, which is why the scale is 4
        and not 3.
        """
        k = self.stage.ctx_scale
        if k == 1:
            return feat
        a, b = self.ctx_off // stride, (self.ctx_off + self.ctx_size) // stride
        return F.interpolate(feat[:, :, a:b, a:b], scale_factor=k,
                             mode="bilinear", align_corners=False)

    # ------------------------------------------------------------ forward ---
    def forward(self, x_ctx, x_det=None, rid=None, n_regions=None,
                apply_region=True):
        """x_ctx: context image (B,C,512,512). x_det: native detail crop, same shape.

        With ctx_scale=1 the two are the same tensor and x_det may be omitted.

        apply_region=False returns the per-pixel distribution with the region layer
        skipped. Tiled inference needs exactly that: cells span the whole drawing, so
        the layer is applied once to the assembled probabilities rather than
        separately inside each 512 tile, where a cell would be cut into pieces. It is
        an explicit argument rather than an inference from rid=None so that
        forgetting rid during training still fails loudly.
        """
        if x_det is None:
            x_det = x_ctx
        f4, f2, f1 = self.semantic(x_ctx)
        f4, f2, f1 = (self._to_detail(f4, 4), self._to_detail(f2, 2),
                      self._to_detail(f1, 1))

        if self.edge is not None:
            e1, e2 = self.edge(x_det)
            g2 = self.fuse2(f2, e2)                                     # stride 2, 64
            up = F.interpolate(g2, size=f1.shape[-2:], mode="bilinear",
                               align_corners=False)
            g1 = self.fuse1(self.pre1(torch.cat([f1, up], 1)), e1)      # stride 1, 32
        else:
            g2 = self.lift2(f2)
            up = F.interpolate(g2, size=f1.shape[-2:], mode="bilinear",
                               align_corners=False)
            g1 = self.pre1(torch.cat([f1, up], 1))

        logits = self.head(g1)
        out = dict(logits=logits,
                   aux=[(self.aux2(g2), self.cfg.w_aux[0]),
                        (self.aux4(f4), self.cfg.w_aux[1])],
                   region=None)

        if self.region is not None and apply_region:
            if rid is None:
                raise ValueError("stage has region_layer=True but no rid was passed")
            r = self.region(logits.float().softmax(1), rid, n_regions)
            out["region"] = r
            out["log_probs"] = r["probs"].clamp_min(1e-7).log()
        else:
            out["log_probs"] = logits.float().log_softmax(1)
        return out

    # --------------------------------------------------------- parameters ---
    def param_groups(self):
        """Pretrained encoder moves at a lower learning rate than the new blocks."""
        enc, rest = [], []
        enc_prefix = ("semantic.stem", "semantic.layer")
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc if name.startswith(enc_prefix) else rest).append(p)
        return enc, rest

    def n_params(self):
        d = {}
        for part in ("semantic", "edge", "region"):
            m = getattr(self, part, None)
            d[part] = 0 if m is None else sum(p.numel() for p in m.parameters())
        d["fusion+head"] = (sum(p.numel() for p in self.parameters())
                            - sum(d.values()))
        d["total"] = sum(p.numel() for p in self.parameters())
        return d


def build(n_classes, stage: Stage, cfg: TrainCfg = TrainCfg()) -> FacadeLineNet:
    return FacadeLineNet(n_classes, stage, cfg)
