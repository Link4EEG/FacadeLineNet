"""Semantic connected components and reproducible image-level indices."""
import math
import numpy as np
from scipy import ndimage
from .palette import load_palette

SEGMENT_COLUMNS=['image','class_id','class_name','segment_id','included_in_indices','area_px','perimeter_px','centroid_x','centroid_y','bbox_x_min','bbox_y_min','bbox_x_max','bbox_y_max','boundary_complexity']
CLASS_COLUMNS=['image','class_id','class_name','included_in_indices','segment_count','total_area_px','min_segment_area_px','max_segment_area_px','mean_segment_area_px','median_segment_area_px','sd_segment_area_px']
IMAGE_COLUMNS=['image','width','height','connectivity','normalisation_scope','normalisation_area_px','architectural_area_px','architectural_area_fraction','N_seg','A_bar_seg','CV_area','L_boundary','C_boundary','H_area','PP_area']

def analyze(labels,image='',connectivity=4,normalisation='image'):
    labels=np.asarray(labels)
    if labels.ndim!=2 or labels.size==0 or labels.dtype.kind not in 'ui' or labels.min()<0 or labels.max()>16:
        raise ValueError('Expected a non-empty 2D integer class-ID mask with IDs 0..16')
    if connectivity not in (4,8): raise ValueError('Connectivity must be 4 or 8')
    if normalisation not in ('image','architectural'): raise ValueError('Normalisation must be image or architectural')
    structure=ndimage.generate_binary_structure(2,1 if connectivity==4 else 2)
    classes=load_palette()['classes']; segments=[]; stats=[]; next_id=1
    segment_map=np.zeros(labels.shape,np.uint32)
    for cls in classes:
        cid=cls['id'];mask=labels==cid; components,n=ndimage.label(mask,structure)
        areas=np.bincount(components.ravel(),minlength=n+1)[1:]
        included=cid not in (0,1)
        stats.append(dict(zip(CLASS_COLUMNS,[image,cid,cls['name'],included,n,int(areas.sum()),
            int(areas.min()) if n else None,int(areas.max()) if n else None,
            float(areas.mean()) if n else None,float(np.median(areas)) if n else None,float(areas.std(ddof=0)) if n else None])))
        if not n: continue
        # Each exposed pixel edge contributes one unit. Includes holes and image edges.
        padded=np.pad(components,1);center=padded[1:-1,1:-1]
        edgecount=np.zeros(labels.shape,np.uint8)
        for neighbor in [padded[:-2,1:-1],padded[2:,1:-1],padded[1:-1,:-2],padded[1:-1,2:]]:
            edgecount+=(center!=neighbor)&(center!=0)
        perimeters=np.bincount(components.ravel(),weights=edgecount.ravel(),minlength=n+1)[1:]
        centroids=ndimage.center_of_mass(mask,components,range(1,n+1))
        boxes=ndimage.find_objects(components)
        for i,(area,perimeter,centroid,box) in enumerate(zip(areas,perimeters,centroids,boxes),1):
            yy,xx=box;cy,cx=centroid;sid=next_id+i-1
            segments.append(dict(zip(SEGMENT_COLUMNS,[image,cid,cls['name'],sid,included,int(area),float(perimeter),
                float(cx+.5),float(cy+.5),xx.start,yy.start,xx.stop,yy.stop,float(perimeter**2/(4*math.pi*area))])))
        segment_map[mask]=components[mask].astype(np.uint32)+next_id-1
        next_id+=n
    selected=[s for s in segments if s['included_in_indices']]
    areas=np.array([s['area_px'] for s in selected],dtype=float)
    architectural=int(areas.sum());denom=labels.size if normalisation=='image' else architectural
    class_areas=np.array([s['total_area_px'] for s in stats if s['included_in_indices']],dtype=float)
    if architectural:
        proportions=class_areas[class_areas>0]/architectural
        entropy=float(-np.sum(proportions*np.log2(proportions)))
    else: entropy=None
    n=len(selected)
    summary=dict(zip(IMAGE_COLUMNS,[image,labels.shape[1],labels.shape[0],connectivity,normalisation,int(denom),
        architectural,architectural/labels.size,n,float(areas.mean()/denom) if n and denom else None,
        float(areas.std(ddof=0)/areas.mean()) if n else None,
        float(sum(s['perimeter_px'] for s in selected)),
        float(np.mean([s['boundary_complexity'] for s in selected])) if n else None,
        entropy,float(2**entropy) if entropy is not None else None]))
    return segments,stats,summary,segment_map
