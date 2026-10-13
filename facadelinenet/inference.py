"""Load an archived A5 checkpoint safely and infer arbitrary native-size drawings."""
from dataclasses import fields
import numpy as np
import torch
from .config import Stage,TrainCfg
from .model import build
from .preprocessing import Drawing
from .tiling import predict_probs
from .modules import region_consistent_numpy

class Predictor:
    def __init__(self,weights,device='auto',precision='auto'):
        if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
        self.device=torch.device(device)
        if self.device.type not in ('cpu','cuda'): raise ValueError('Supported devices: cpu or cuda')
        ck=torch.load(weights,map_location='cpu',weights_only=True)
        if ck.get('stage')!='A5' or ck.get('n_classes')!=17: raise ValueError('Expected a 17-class A5 checkpoint')
        config={k:v for k,v in ck.get('cfg',{}).items() if k in {f.name for f in fields(TrainCfg)}}
        config['encoder_weights']='none'
        self.stage=Stage(); self.model=build(17,self.stage,TrainCfg(**config))
        self.model.load_state_dict(ck['model'],strict=True)
        self.model=self.model.to(self.device).to(memory_format=torch.channels_last).eval()
        if precision=='auto': precision=('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if self.device.type=='cuda' else 'fp32'
        if self.device.type=='cpu' and precision!='fp32': raise ValueError('CPU inference requires fp32')
        self.dtype={'fp32':None,'bf16':torch.bfloat16,'fp16':torch.float16}[precision]
        self.precision=precision; self.seed=ck.get('cfg',{}).get('seed'); self.iteration=ck.get('iter')
    def predict(self,path):
        draw=Drawing(path)
        with torch.inference_mode():
            probs=predict_probs(self.model,draw,self.device,self.stage,self.dtype)
        probs=region_consistent_numpy(probs.astype(np.float32),draw.rid,int(draw.rid.max())+1,
                    float(self.model.region.tau.detach().cpu()),float(self.model.region.temp.detach().cpu()))
        labels=probs.argmax(0).astype(np.uint8)
        labels[draw.rid==0]=1
        return draw.rgb,labels
