from dataclasses import dataclass
CROP=512
STRIDE=384
def ctx_geometry(crop=CROP,ctx_scale=4):
    window=crop*ctx_scale
    return window,((window-crop)//2)//ctx_scale,crop//ctx_scale
@dataclass
class Stage:
    name: str='A5'
    ctx_scale: int=4
    edge_branch: bool=True
    aux_channels: bool=True
    region_layer: bool=True
    in_channels: int=3
@dataclass
class TrainCfg:
    encoder_weights: str='none'
    decoder_channels: tuple=(256,128,64,32,16)
    aspp_channels: int=256
    aspp_rates: tuple=(6,12,18)
    edge_channels: tuple=(32,64,128)
    dropout: float=0.1
    w_aux: tuple=(0.4,0.2)
