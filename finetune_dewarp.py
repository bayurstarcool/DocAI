import os, sys, json, time, random, glob, zipfile, gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

DOCRES_DIR = Path("/home/wahyu/DocRes")
sys.path.insert(0, str(DOCRES_DIR))
from models import restormer_arch
from utils import convert_state_dict

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
USE_AMP = os.environ.get("DEWARP_AMP", "1") == "1"
BATCH_SIZE = int(os.environ.get("DEWARP_BATCH_SIZE", "1"))
LR = float(os.environ.get("DEWARP_LR", "1e-5"))
NUM_EPOCHS = int(os.environ.get("DEWARP_EPOCHS", "30"))
IM_SIZE = int(os.environ.get("DEWARP_IM_SIZE", "512"))
NUM_WORKERS = 0
MAX_ITERS = int(os.environ.get("DEWARP_MAX_ITERS", "0"))
SAVE_EVERY = int(os.environ.get("DEWARP_SAVE_EVERY", "0"))
OUT_DIR = Path("/home/wahyu/docai/checkpoints")
OUT_DIR.mkdir(exist_ok=True)
RUN_TAG = os.environ.get("DEWARP_RUN_TAG", time.strftime("dewarp_geom_%Y%m%d_%H%M%S"))
LOG_DIR = Path("/home/wahyu/DocRes/finetune_logs") / RUN_TAG
LOG_DIR.mkdir(parents=True, exist_ok=True)

def random_warp(img, strength=0.6):
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wt = random.choice(["fold", "curl", "crease", "perspective", "combined"])
    if wt in ("fold","combined"):
        axis = random.choice(["h","v"])
        fp = int((h if axis=="h" else w) * random.uniform(0.3,0.7))
        amp = min(w,h)*strength*random.uniform(0.08,0.2)
        if axis=="h":
            off = np.where(grid_y<fp, amp*(grid_y-fp)/fp, amp*(grid_y-fp)/(h-fp))
            grid_x += off
        else:
            off = np.where(grid_x<fp, amp*(grid_x-fp)/fp, amp*(grid_x-fp)/(w-fp))
            grid_y += off
    if wt in ("curl","combined"):
        corner = random.choice(["tl","tr","bl","br"])
        cs = min(w,h)*strength*random.uniform(0.15,0.35)
        if corner=="tl": d=np.sqrt(grid_x**2+grid_y**2); sx,sy=-1,-1
        elif corner=="tr": d=np.sqrt((w-1-grid_x)**2+grid_y**2); sx,sy=1,-1
        elif corner=="bl": d=np.sqrt(grid_x**2+(h-1-grid_y)**2); sx,sy=-1,1
        else: d=np.sqrt((w-1-grid_x)**2+(h-1-grid_y)**2); sx,sy=1,1
        m=d<cs; o=(cs-d[m])*random.uniform(0.3,0.7)
        grid_x[m]+=o*sx; grid_y[m]+=o*sy
    if wt in ("crease","combined"):
        for _ in range(random.randint(1,3)):
            ac=min(w,h)*strength*0.03
            cx,cy=random.uniform(0.2,0.8)*w,random.uniform(0.2,0.8)*h
            ca=random.uniform(0,np.pi)
            d=(grid_x-cx)*np.cos(ca)+(grid_y-cy)*np.sin(ca)
            o=ac*np.exp(-d**2/(w*0.03)**2)*np.sin(d*0.5)
            grid_x+=o*np.cos(ca+np.pi/2); grid_y+=o*np.sin(ca+np.pi/2)
    if wt in ("perspective","combined"):
        margin=min(w,h)*strength*0.12
        src=np.float32([[0,0],[w,0],[0,h],[w,h]])
        dst=np.float32([[random.uniform(-margin,margin),random.uniform(-margin,margin)],[w+random.uniform(-margin,margin),random.uniform(-margin,margin)],[random.uniform(-margin,margin),h+random.uniform(-margin,margin)],[w+random.uniform(-margin,margin),h+random.uniform(-margin,margin)]])
        M=cv2.getPerspectiveTransform(dst,src)
        pts=np.stack([grid_x,grid_y],-1).reshape(-1,1,2).astype(np.float32)
        inv=cv2.perspectiveTransform(pts,M).reshape(h,w,2)
        grid_x,grid_y=inv[:,:,0],inv[:,:,1]
    np.clip(grid_x,0,w-1,out=grid_x); np.clip(grid_y,0,h-1,out=grid_y)
    warped=cv2.remap(img,grid_x,grid_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    bm=np.stack([grid_x/w,grid_y/h],-1).astype(np.float32)
    return warped,bm

def build_pairs():
    pairs=[]
    inv3d_dir=Path("/home/wahyu/docai/datasets/inv3d")
    if inv3d_dir.exists():
        targets={f.stem:str(f) for f in (inv3d_dir/"target").glob("*.png")}
        for f in (inv3d_dir/"input").glob("*.*"):
            doc_id=f.name.split("_")[0]
            if doc_id in targets:
                setting = f.stem.split("_")[-1].lower()
                # Geometry-only training: remove shadow from train by default.
                shadow_train = os.environ.get("DEWARP_SHADOW_TRAIN", "0") == "1"
                if setting in ("color", "bright") or (setting == "shadow" and shadow_train):
                    pairs.append({"in":str(f),"gt":targets[doc_id],"src":"inv3d","setting":setting})
                else:
                    pairs.append({"in":str(f),"gt":targets[doc_id],"src":"inv3d_shadow_eval","setting":setting})
    inv3dr_dir=Path("/home/wahyu/docai/datasets/inv3dreal")
    if inv3dr_dir.exists():
        targets={f.stem.replace("_flat",""):str(f) for f in inv3dr_dir.glob("*flat*")}
        for f in inv3dr_dir.glob("*warped*"):
            key=f.stem.replace("_warped","")
            if key in targets:
                pairs.append({"in":str(f),"gt":targets[key],"src":"inv3dreal"})
    custom_targets=list(set(glob.glob("/home/wahyu/docai/datasets/paired/*/target/*.png")+glob.glob("/home/wahyu/docai/datasets/paired/*/custom/target/*.png"))) if os.environ.get("DEWARP_SYNTH","1")=="1" else []
    pairs.append({"custom_clean":custom_targets,"src":"synthetic"})
    print(f"Inv3D train pairs: {len([p for p in pairs if p.get('src') in ('inv3d','inv3dreal')])}, Shadow eval pairs: {len([p for p in pairs if p.get('src') == 'inv3d_shadow_eval'])}, Clean sources: {len(custom_targets)}")
    return pairs

class DewarpDataset(Dataset):
    def __init__(self,pairs,split="train",ratio=0.9,augment=True):
        self.inv3d_pairs=[p for p in pairs if p.get("src") in ("inv3d","inv3dreal")]
        self.clean_sources=[p for p in pairs if p.get("src")=="synthetic"][0].get("custom_clean",[])
        random.shuffle(self.inv3d_pairs)
        si=int(len(self.inv3d_pairs)*ratio)
        self.inv3d_pairs=self.inv3d_pairs[:si] if split=="train" else self.inv3d_pairs[si:]
        self.augment=augment and split=="train"
        self.synth_per_epoch=max(2000,len(self.inv3d_pairs)*3)
    def __len__(self):
        return len(self.inv3d_pairs)+(self.synth_per_epoch if self.augment else 0)
    def __getitem__(self,idx):
        if idx<len(self.inv3d_pairs):
            p=self.inv3d_pairs[idx]
            in_im=cv2.imread(p["in"])
            gt=cv2.imread(p["gt"])
            if in_im is None or gt is None:
                in_im,gt,bm=self._make_synthetic()
            else:
                if in_im.shape[:2]!=gt.shape[:2]:
                    gt=cv2.resize(gt,(in_im.shape[1],in_im.shape[0]))
                h,w=in_im.shape[:2]
                f=cv2.calcOpticalFlowFarneback(cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY),cv2.cvtColor(in_im,cv2.COLOR_BGR2GRAY),None,0.5,3,15,3,5,1.2,0)
                gx,gy=np.meshgrid(np.arange(w),np.arange(h))
                sx,sy=gx+f[:,:,0],gy+f[:,:,1]
                np.clip(sx,0,w-1,out=sx); np.clip(sy,0,h-1,out=sy)
                bm=np.stack([sx/w,sy/h],-1).astype(np.float32)
        else:
            in_im,gt,bm=self._make_synthetic()
        h,w=in_im.shape[:2]
        in_r=cv2.resize(in_im,(IM_SIZE,IM_SIZE))
        bm_r=cv2.resize(bm,(IM_SIZE,IM_SIZE))
        in_t=torch.from_numpy(in_r.transpose(2,0,1).astype(np.float32)/255.0)
        c0=np.tile(np.arange(IM_SIZE).reshape(IM_SIZE,1),(1,IM_SIZE)).astype(np.float32)/IM_SIZE
        c1=np.tile(np.arange(IM_SIZE).reshape(1,IM_SIZE),(IM_SIZE,1)).astype(np.float32)/IM_SIZE
        coord=np.stack([c1,c0],-1)
        prompt=np.concatenate([coord,np.ones((IM_SIZE,IM_SIZE,1),dtype=np.float32)],-1).transpose(2,0,1)
        x=torch.cat([in_t,torch.from_numpy(prompt)],dim=0)
        y=torch.from_numpy(bm_r.transpose(2,0,1).astype(np.float32))
        if self.augment:
            if random.random()>0.5:
                x=torch.flip(x,dims=[2]); y=torch.flip(y,dims=[2]); y[0]=1-y[0]
            if random.random()>0.5:
                x=torch.flip(x,dims=[1]); y=torch.flip(y,dims=[1]); y[1]=1-y[1]
        return x,y
    def _make_synthetic(self):
        src=random.choice(self.clean_sources)
        gt=cv2.imread(src)
        if gt is None or min(gt.shape[:2])<200:
            gt=np.zeros((512,512,3),dtype=np.uint8)+255
        h,w=gt.shape[:2]
        if max(w,h)>1024:
            s=1024/max(w,h)
            gt=cv2.resize(gt,(int(w*s),int(h*s)))
        warped, bm = random_warp(gt); return warped, gt, bm

def smooth_loss(pred):
    dx=torch.abs(pred[:,:,:,:-1]-pred[:,:,:,1:]).mean()
    dy=torch.abs(pred[:,:,:-1,:]-pred[:,:,1:,:]).mean()
    return dx+dy

def bm_loss(pred,target):
    l1=F.l1_loss(pred,target)
    return l1+0.02*smooth_loss(pred)+0.02*smooth_loss(target)

model=restormer_arch.Restormer(inp_channels=6,out_channels=3,dim=48,num_blocks=[2,3,3,4],num_refinement_blocks=4,heads=[1,2,4,8],ffn_expansion_factor=2.66,bias=False,LayerNorm_type="WithBias",dual_pixel_task=True)
print("Loading pretrained checkpoint...")
state=convert_state_dict(torch.load(str(DOCRES_DIR/"checkpoints"/"docres.pkl"),map_location="cpu")["model_state"])
model.load_state_dict(state,strict=False)
model.to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

print("Building dataset...")
all_pairs=build_pairs()
train_ds=DewarpDataset(all_pairs,"train")
val_ds=DewarpDataset(all_pairs,"val",augment=False)
train_loader=DataLoader(train_ds,BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True)
if os.environ.get("DEWARP_CACHE_DATA","0")=="1" and len(train_ds):
    _ = train_ds[0]
val_loader=DataLoader(val_ds,BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=NUM_EPOCHS)
best_val=float("inf")
log_file=open(LOG_DIR/"log.txt","a")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss,t0=0,time.time()
    for batch_idx,(x,y) in enumerate(tqdm(train_loader,desc=f"E{epoch+1}",leave=False), 1):
        x,y=x.to(DEVICE),y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            pred=model(x)
            loss=bm_loss(pred[:,:2].float(),y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        loss_item = loss.item()
        train_loss+=loss_item
        if SAVE_EVERY and batch_idx % SAVE_EVERY == 0:
            tmp_name = os.environ.get("DEWARP_CKPT_NAME", "dewarp_finetune_best.pth")
            stem = Path(tmp_name).stem
            ckpt_path = OUT_DIR / f"{stem}_iter{batch_idx}.pth"
            torch.save({"model_state":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"iter":batch_idx,"val_loss":None,"im_size":IM_SIZE,"max_iters":MAX_ITERS},str(ckpt_path))
            print(f"  Saved interim {ckpt_path.name} at iter {batch_idx}")
        del pred, loss
        if MAX_ITERS and batch_idx >= MAX_ITERS:
            break
    model.eval()
    val_loss=0
    with torch.no_grad():
        for x,y in val_loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                pred=model(x)
                vloss=bm_loss(pred[:,:2].float(),y.float())
            val_loss+=vloss.item()
            del pred, vloss
    train_loss/= (MAX_ITERS if MAX_ITERS else len(train_loader))
    val_loss/=len(val_loader)
    scheduler.step()
    lr_now=scheduler.get_last_lr()[0]
    elapsed=time.time()-t0
    msg=f"E{epoch+1:2d}/{NUM_EPOCHS} train={train_loss:.6f} val={val_loss:.6f} lr={lr_now:.2e} {elapsed:.0f}s"
    print(msg)
    log_file.write(msg+"\n"); log_file.flush()
    if val_loss<best_val:
        best_val=val_loss
        ckpt_name = os.environ.get("DEWARP_CKPT_NAME", "dewarp_finetune_best.pth")
        torch.save({"model_state":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"val_loss":val_loss,"im_size":IM_SIZE,"max_iters":MAX_ITERS},str(OUT_DIR/ckpt_name))
        print(f"  Saved best {ckpt_name} (val_loss={val_loss:.6f})")

log_file.close()
print(f"Done! Best: {OUT_DIR / 'dewarp_finetune_best.pth'}")
