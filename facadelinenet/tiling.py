import numpy as np
import torch
from . import preprocessing as D
from .config import CROP, STRIDE

def tile_starts(n, tile, stride, align=1):
    """Tile origins covering [0, n), all multiples of `align`.

    The regular positions are multiples of the stride, and 384 is already a multiple
    of the context scale. Only the last tile is awkward: n - tile is generally not on
    the grid. Rounding it UP keeps the origin aligned and still covers the final row,
    because the tile then runs past the edge and Drawing.window pads it as page. (Down
    would leave up to align-1 rows unwritten, and leaving it unaligned would shift the
    context window against the detail crop on exactly the edge tiles.)
    """
    xs = list(range(0, max(n - tile, 0) + 1, stride))
    last = max(0, n - tile)
    if align > 1:
        last = -(-last // align) * align
    if not xs or xs[-1] < last:
        xs.append(last)
    return xs

def predict_probs(model, draw: D.Drawing, device, stage, amp_dtype,
                  tile=CROP, stride=STRIDE):
    """Assemble the whole drawing's class probabilities at native resolution."""
    import torch

    H, W, K = draw.H, draw.W, model.n_classes
    out = np.zeros((K, H, W), np.float16)
    m = (tile - stride) // 2
    window, _, _ = D.ctx_geometry(tile, stage.ctx_scale)

    align = stage.ctx_scale
    ys = tile_starts(H, tile, stride, align)
    xs = tile_starts(W, tile, stride, align)

    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                det = D.channels(draw, y0, x0, tile, stage.aux_channels)
                if stage.ctx_scale == 1:
                    ctx = det
                else:
                    cy = y0 + tile // 2 - window // 2
                    cx = x0 + tile // 2 - window // 2
                    ctx = D.channels(draw, cy, cx, window, stage.aux_channels,
                                     k=stage.ctx_scale)
                td = torch.from_numpy(det[None]).to(
                    device).to(memory_format=torch.channels_last)
                tc = torch.from_numpy(np.ascontiguousarray(ctx)[None]).to(
                    device).to(memory_format=torch.channels_last)
                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda" and amp_dtype is not None):
                    # raw pixel probabilities: the region layer runs once, later, on
                    # the assembled drawing, where the cells are whole
                    logit = model(tc, td, apply_region=False)["logits"]
                p = logit.float().softmax(1)[0].to("cpu").numpy().astype(np.float16)

                ty0 = 0 if y0 == 0 else m
                tx0 = 0 if x0 == 0 else m
                ty1 = tile if y0 + tile >= H else tile - m
                tx1 = tile if x0 + tile >= W else tile - m
                gy0, gx0 = y0 + ty0, x0 + tx0
                gy1, gx1 = min(y0 + ty1, H), min(x0 + tx1, W)
                out[:, gy0:gy1, gx0:gx1] = p[:, ty0:ty0 + (gy1 - gy0),
                                             tx0:tx0 + (gx1 - gx0)]
    return out
