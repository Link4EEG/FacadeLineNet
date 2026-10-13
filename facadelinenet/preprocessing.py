"""Input-only preprocessing; no ground truth or project cache is required."""
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import derive,_pyramid,channels
from .config import ctx_geometry

def read_rgb(path):
    with Image.open(path) as im:
        if 'A' in im.getbands() or 'transparency' in im.info:
            rgba=im.convert('RGBA'); base=Image.new('RGBA',im.size,'white')
            im=Image.alpha_composite(base,rgba)
        return im.convert('RGB')

class Drawing:
    PAD={'gray':255,'dist':255,'ach':255,'rid':0}
    def __init__(self,path):
        self.rgb=read_rgb(path)
        self.gray=np.asarray(self.rgb.convert('L'),dtype=np.uint8)
        self.H,self.W=self.gray.shape
        _,self.dist,self.rid,self.area,self.ach=derive(self.gray)
        self.arr={(key,1):getattr(self,key) for key in self.PAD}
        for key in ['gray','dist','ach']:
            self.arr[key,4]=_pyramid(self.arr[key,1],4,self.PAD[key])
    def has_pyramid(self,k): return ("gray",k) in self.arr
    def window(self,key,y0,x0,size,level=1):
        src=self.arr[key,level]
        if level!=1:
            if y0%level or x0%level or size%level: raise ValueError('Unaligned context window')
            y0,x0,size=y0//level,x0//level,size//level
        h,w=src.shape;y1,x1=y0+size,x0+size
        sy0,sx0,sy1,sx1=max(y0,0),max(x0,0),min(y1,h),min(x1,w)
        if (sy0,sx0,sy1,sx1)==(y0,x0,y1,x1): return np.asarray(src[y0:y1,x0:x1])
        out=np.full((size,size),self.PAD[key]*level*level,src.dtype)
        if sy0<sy1 and sx0<sx1: out[sy0-y0:sy1-y0,sx0-x0:sx1-x0]=src[sy0:sy1,sx0:sx1]
        return out
