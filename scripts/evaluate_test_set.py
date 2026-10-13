"""Score exported class-ID PNGs against native RGB expert masks using the paper metric."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
from facadelinenet.metrics import expert_labels,image_miou
def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--ground-truth',type=Path,required=True)
    p.add_argument('--split',type=Path,default=Path('metadata/test_split.txt'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    files=a.split.read_text().splitlines();results=[]
    if len(files)!=len(set(files)):raise ValueError('Duplicate split entries')
    for name in files:
        pp=a.predictions/(name+'.png')
        if not pp.exists():pp=a.predictions/name
        with Image.open(pp) as im:pred=np.asarray(im)
        with Image.open(a.ground_truth/name) as im:gt=expert_labels(im.convert('RGB'))
        score=image_miou(pred,gt)
        if score is None:raise ValueError(f'No valid GT pixels: {name}')
        results.append([name,score])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['Image','mIoU']);w.writerows(results)
    print(json.dumps({'n':len(results),'mean_image_miou':float(np.mean([r[1] for r in results]))}))
if __name__=='__main__':main()
