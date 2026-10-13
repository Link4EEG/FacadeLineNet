import argparse,csv,json,hashlib,platform,time
from pathlib import Path
import numpy as np
from PIL import Image
from .analysis import analyze,SEGMENT_COLUMNS,CLASS_COLUMNS,IMAGE_COLUMNS
from .palette import colorize
from . import __version__

def write_csv(path,columns,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def main(argv=None):
    p=argparse.ArgumentParser(description='Native-resolution A5 segmentation and semantic structure indices')
    p.add_argument('--input',required=True,type=Path,help='One image or directory (recursive)')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--weights',type=Path,default=Path('weights/A5_seed0.pt'))
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--precision',choices=['auto','fp32','fp16','bf16'],default='auto')
    p.add_argument('--connectivity',type=int,choices=[4,8],default=4)
    p.add_argument('--normalisation',choices=['image','architectural'],default='image')
    p.add_argument('--overwrite',action='store_true',help='Allow replacement of previous generated results')
    p.add_argument('--labels-only',action='store_true',help='Analyze existing uint8 class-ID PNGs without inference')
    args=p.parse_args(argv)
    if not args.input.exists(): p.error('Input does not exist')
    root=args.input if args.input.is_dir() else args.input.parent
    extensions={'.png'} if args.labels_only else {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
    files=sorted((f for f in root.rglob('*') if f.is_file() and f.suffix.lower() in extensions and not f.name.startswith('._')),key=lambda f:f.relative_to(root).as_posix()) if args.input.is_dir() else [args.input]
    if not files: p.error('No supported images found')
    if args.output.resolve()==root.resolve() or root.resolve() in args.output.resolve().parents: p.error('Output must be outside the input directory')
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite: p.error('Output is non-empty; choose another folder or use --overwrite')
    predictor=None
    if not args.labels_only:
        if not args.weights.is_file(): p.error('Checkpoint missing. Run python scripts/download_weights.py --seed 0')
        from .inference import Predictor
        predictor=Predictor(args.weights,args.device,args.precision)
    for sub in ['segmentations','class_ids','overlays','segment_ids']:(args.output/sub).mkdir(parents=True,exist_ok=True)
    allseg=[];allclass=[];allimage=[];provenance=[]
    for file in files:
        relative=file.relative_to(root).as_posix(); start=time.monotonic()
        if args.labels_only:
            with Image.open(file) as im: labels=np.asarray(im)
            rgb=None
        else: rgb,labels=predictor.predict(file)
        segments,classes,summary,ids=analyze(labels,relative,args.connectivity,args.normalisation)
        # Retain the source extension to prevent same-stem PNG/JPEG collisions.
        dest=Path(relative+'.png')
        for sub in ['segmentations','class_ids','overlays','segment_ids']:(args.output/sub/dest).parent.mkdir(parents=True,exist_ok=True)
        colored=Image.fromarray(colorize(labels))
        colored.save(args.output/'segmentations'/dest)
        Image.fromarray(labels.astype(np.uint8)).save(args.output/'class_ids'/dest)
        np.save((args.output/'segment_ids'/dest).with_suffix('.npy'),ids)
        if rgb is not None: Image.blend(rgb,colored,0.45).save(args.output/'overlays'/dest)
        allseg.extend(segments);allclass.extend(classes);allimage.append(summary)
        provenance.append({'image':relative,'sha256':sha256(file),'width':labels.shape[1],'height':labels.shape[0],'seconds':time.monotonic()-start})
        print(f'{relative}: {summary["N_seg"]} architectural segments',flush=True)
    write_csv(args.output/'facade_segments.csv',SEGMENT_COLUMNS,allseg)
    write_csv(args.output/'facade_class_summary.csv',CLASS_COLUMNS,allclass)
    write_csv(args.output/'facade_segmentation_analysis_results.csv',IMAGE_COLUMNS,allimage)
    metadata={'version':__version__,'python':platform.python_version(),'mode':'labels-only' if args.labels_only else 'A5',
        'connectivity':args.connectivity,'normalisation':args.normalisation,'excluded_image_index_classes':[0,1],
        'area_sd_ddof':0,'perimeter':'exposed 4-neighbour unit pixel edges; holes included; shared interfaces counted for both segments',
        'centroids':'pixel centres x+0.5,y+0.5; origin top-left','bounding_boxes':'zero-based [xmin,ymin,xmax,ymax), maxima exclusive',
        'empty_architectural_mask':'N_seg and L_boundary=0; undefined means/CV/entropy/perplexity=null (blank CSV)',
        'images':provenance}
    if predictor:
        import torch,torchvision
        metadata.update(weights_filename=args.weights.name,weights_sha256=sha256(args.weights),seed=predictor.seed,checkpoint_iteration=predictor.iteration,
            device=str(predictor.device),precision=predictor.precision,torch=torch.__version__,torchvision=torchvision.__version__)
    (args.output/'run_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    return 0
if __name__=='__main__': raise SystemExit(main())
