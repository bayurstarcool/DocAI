from __future__ import annotations
from pathlib import Path
import random, cv2, numpy as np, torch
from torch.utils.data import Dataset
from .utils import list_images, imread_rgb
from .synth_shadow import apply_document_shadow

class CleanDocSyntheticShadowDataset(Dataset):
    def __init__(self, root, size=768, augment=True):
        self.paths=list_images(root)
        if not self.paths: raise RuntimeError(f'No images found: {root}')
        self.size=size; self.augment=augment
    def __len__(self): return len(self.paths)
    def _resize_crop(self,img):
        h,w=img.shape[:2]; s=self.size
        scale=max(s/h,s/w) if self.augment else min(s/h,s/w)
        nh,nw=int(h*scale+.5),int(w*scale+.5)
        img=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
        if self.augment:
            y=random.randint(0,max(0,nh-s)); x=random.randint(0,max(0,nw-s))
            img=img[y:y+s,x:x+s]
        else:
            canvas=np.ones((s,s,3),np.uint8)*255; y=(s-nh)//2; x=(s-nw)//2; canvas[y:y+nh,x:x+nw]=img; img=canvas
        return img
    def __getitem__(self,i):
        clean=imread_rgb(self.paths[i])
        clean=self._resize_crop(clean)
        if self.augment and random.random()<.5: clean=cv2.rotate(clean, random.choice([cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_90_COUNTERCLOCKWISE,cv2.ROTATE_180]))
        shadow,mask=apply_document_shadow(clean, True)
        x=torch.from_numpy(shadow.transpose(2,0,1)).float()/255.
        y=torch.from_numpy(clean.transpose(2,0,1)).float()/255.
        m=torch.from_numpy(mask[None]).float()/255.
        return {'input':x,'target':y,'mask':m,'path':str(self.paths[i])}

class PairedShadowDataset(Dataset):
    def __init__(self, shadow_dir, clean_dir, mask_dir=None, size=1024):
        shadow_dirs=[Path(p) for p in str(shadow_dir).split(',') if p]
        clean_dirs=[Path(p) for p in str(clean_dir).split(',') if p]
        mask_dirs=[Path(p) for p in str(mask_dir).split(',') if p] if mask_dir else []
        if len(clean_dirs) != len(shadow_dirs):
            raise RuntimeError('paired shadow/clean directory counts differ')
        if mask_dirs and len(mask_dirs) != len(shadow_dirs):
            raise RuntimeError('paired shadow/mask directory counts differ')
        self.items=[]; self.size=size
        for index,(sdir,cdir) in enumerate(zip(shadow_dirs,clean_dirs)):
            mdir=mask_dirs[index] if mask_dirs else None
            for sp in list_images(sdir):
                cp=cdir/sp.name
                if not cp.exists(): cp=cdir/(sp.stem+'.png')
                if cp.exists(): self.items.append((sp,cp,mdir))
        if not self.items: raise RuntimeError(f'No paired images: {shadow_dir}')
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        sp,cp,mask_dir=self.items[i]
        x=imread_rgb(sp); y=imread_rgb(cp)
        x=cv2.resize(x,(self.size,self.size)); y=cv2.resize(y,(self.size,self.size))
        if mask_dir:
            mp=mask_dir/sp.name
            m=cv2.imread(str(mp),0) if mp.exists() else np.zeros(x.shape[:2],np.uint8)
            m=cv2.resize(m,(self.size,self.size))
        else:
            m=np.zeros(x.shape[:2],np.uint8)
        return {'input':torch.from_numpy(x.transpose(2,0,1)).float()/255., 'target':torch.from_numpy(y.transpose(2,0,1)).float()/255., 'mask':torch.from_numpy(m[None]).float()/255., 'path':str(sp)}
