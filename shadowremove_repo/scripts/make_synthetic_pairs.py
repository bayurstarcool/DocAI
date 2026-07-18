from __future__ import annotations
import argparse, shutil
from pathlib import Path
from tqdm import tqdm
from src.utils import list_images, imread_rgb, imwrite_rgb
from src.synth_shadow import apply_document_shadow
p=argparse.ArgumentParser(); p.add_argument('--clean',required=True); p.add_argument('--out',default='data/synth_pairs'); p.add_argument('--copies',type=int,default=5); a=p.parse_args()
out=Path(a.out); (out/'shadow').mkdir(parents=True,exist_ok=True); (out/'clean').mkdir(parents=True,exist_ok=True); (out/'mask').mkdir(parents=True,exist_ok=True)
for pth in tqdm(list_images(a.clean)):
    clean=imread_rgb(pth)
    for i in range(a.copies):
        shadow,mask=apply_document_shadow(clean,True); name=f'{pth.stem}_{i:03d}.jpg'
        imwrite_rgb(out/'shadow'/name,shadow); imwrite_rgb(out/'clean'/name,clean); imwrite_rgb(out/'mask'/name,mask)
print('saved to',out)
