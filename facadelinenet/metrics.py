"""Manuscript image-wise GT-present mIoU, separate from structural indices."""
import numpy as np
from .palette import load_palette
def expert_labels(rgb):
    pal=load_palette();colors=np.array([c['rgb'] for c in pal['classes']],np.int16)
    a=np.asarray(rgb,dtype=np.int16);flat=a.reshape(-1,3);out=np.empty(len(flat),np.uint8)
    for start in range(0,len(flat),100000):
        distances=np.abs(flat[start:start+100000,None,:]-colors).max(axis=2)
        idx=distances.argmin(axis=1);minimum=distances[np.arange(len(idx)),idx]
        out[start:start+len(idx)]=np.where(minimum<=pal['snap_tolerance'],idx,255)
    return out.reshape(a.shape[:2])
def image_miou(pred,gt):
    pred,gt=np.asarray(pred),np.asarray(gt)
    if pred.shape!=gt.shape or pred.ndim!=2: raise ValueError('Prediction and GT must share native H x W')
    if not np.isin(pred,np.arange(17)).all() or not np.isin(gt,list(range(17))+[255]).all(): raise ValueError('Invalid class ID')
    valid=gt!=255
    if not valid.any(): return None
    cm=np.bincount(gt[valid].astype(np.int64)*17+pred[valid],minlength=289).reshape(17,17)
    support=cm.sum(1);union=support+cm.sum(0)-cm.diagonal()
    return float(np.mean(cm.diagonal()[support>0]/union[support>0]))
