from __future__ import annotations
import numpy as np
from scipy import ndimage
INK_THRESH=128
DIST_CAP=64.0

def derive(gray: np.ndarray):
    """(ink, dist_uint8, rid_int32, area_float32) from one grayscale drawing."""
    ink = gray < INK_THRESH
    if ink.mean() < 0.0005:              # 239_1997_2223 is drawn in grey, not black
        ink = gray < 200
    dist = ndimage.distance_transform_edt(~ink)
    dist = np.log1p(np.minimum(dist, DIST_CAP)) / np.log1p(DIST_CAP)
    dist_u8 = np.rint(dist * 255).astype(np.uint8)

    lab, n = ndimage.label(~ink)          # 4-connectivity, as region_baseline.py
    rid = lab.astype(np.int32)            # 0 is ink by construction
    counts = np.bincount(rid.ravel(), minlength=n + 1).astype(np.float64)
    counts[0] = 0.0
    area = (np.log1p(counts) / np.log1p(gray.size)).astype(np.float32)
    # The per-pixel area channel is stored rather than looked up per crop. The
    # lookup is a 4.2 M-element fancy index on every context window, which measured
    # at 20 ms - about as much as the whole GPU step it would be feeding.
    ach = np.rint(area[rid] * 255).astype(np.uint8)
    return ink, dist_u8, rid, area, ach

def _pyramid(a, k, pad):
    """Block SUMS by k, after padding the ragged edge with the array's pad value.

    Sums, not means: 255*16 = 4080 fits in uint16, so the reduction stays exact and
    the pyramid path produces bit-identical values to reducing the full-resolution
    window. Rounding to uint8 here would cost about 0.004 in normalised units and
    quietly break the alignment identity the tests assert.

    The padding matters for the same reason. Trimming the last H % k rows instead
    would make the bottom and right block rows disagree with the full-resolution
    path by up to one pad-vs-pixel difference - measured at 8e-3, small enough to
    pass unnoticed and large enough to mean the two branches disagree about the
    edge of the drawing.
    """
    h = -(-a.shape[0] // k) * k
    w = -(-a.shape[1] // k) * k
    if (h, w) != a.shape:
        full = np.full((h, w), pad, a.dtype)
        full[:a.shape[0], :a.shape[1]] = a
        a = full
    return a.reshape(h // k, k, w // k, k).sum(axis=(1, 3), dtype=np.uint16)

def _down_u8(a, k):
    """downscale() for uint8, summing in uint16 instead of averaging in float64.

    numpy promotes uint8 to float64 for .mean(), which on a 2048 window is 34 MB of
    temporaries per channel. 255*16 fits in uint16, so the sum is exact and the
    single divide happens once, in float32.
    """
    if k == 1:
        return a.astype(np.float32)
    c = a.shape[-2] // k
    s = a.reshape(*a.shape[:-2], c, k, c, k).sum(axis=(-3, -1), dtype=np.uint16)
    return s.astype(np.float32) * (1.0 / (k * k))

def channels(draw: Drawing, y0, x0, size, aux: bool, k: int = 1):
    """(C, size/k, size/k) float32 in [-1, 1], area-averaged by k on the way out.

    Reduction happens per source array, before the channels are stacked. Building the
    3-channel tensor first and reducing it afterwards does the same arithmetic k*k
    times over, and three times over again for the grayscale-replicated case.

    When the cache holds a pyramid at level k the reduction is already done and this
    is a plain windowed read; otherwise it falls back to reducing the full-resolution
    window, which gives bit-identical values at 16x the memory traffic.
    """
    pyr = k != 1 and draw.has_pyramid(k)
    scale = 1.0 / (255.0 * (k * k if pyr else 1))

    def band(key):
        if pyr:
            return draw.window(key, y0, x0, size, level=k).astype(np.float32) * scale
        return _down_u8(draw.window(key, y0, x0, size), k) * scale

    g = band("gray")
    if not aux:
        out = np.empty((3,) + g.shape, np.float32)
        out[:] = g                                   # broadcast, one copy per channel
    else:
        out = np.stack([g, band("dist"), band("ach")])
    out *= 2.0
    out -= 1.0
    return out
