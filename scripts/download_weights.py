"""Download a version-pinned release checkpoint and verify SHA-256."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    p.add_argument('--output',type=Path,default=Path('weights'));a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    row=next(r for r in json.loads((root/'metadata/weights_manifest.json').read_text())['models'] if r['seed']==a.seed)
    a.output.mkdir(parents=True,exist_ok=True);dest=a.output/row['filename']
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==row['sha256']: print(f'Already verified: {dest}');return
    temp=dest.with_suffix('.download')
    request=urllib.request.Request(row['url'],headers={'User-Agent':'FacadeLineNet/1.0.1'})
    with urllib.request.urlopen(request,timeout=120) as src,temp.open('wb') as out:
        while chunk:=src.read(1024*1024):out.write(chunk)
    if hashlib.sha256(temp.read_bytes()).hexdigest()!=row['sha256']: raise RuntimeError('Checkpoint hash mismatch; download not installed')
    temp.replace(dest);print(f'Downloaded and verified: {dest}')
if __name__=='__main__':main()
