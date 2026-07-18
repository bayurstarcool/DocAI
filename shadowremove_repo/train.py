from __future__ import annotations
import argparse, os, yaml, time
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.dataset import CleanDocSyntheticShadowDataset, PairedShadowDataset
from src.model import DocEnhancerNet
from src.utils import seed_everything, imwrite_rgb, psnr, ssim_rgb

try:
    from torch.amp import GradScaler, autocast
except Exception:
    from torch.cuda.amp import GradScaler, autocast

def tv_loss(x): return (x[:,:,1:]-x[:,:,:-1]).abs().mean() + (x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean()
def charbonnier(a,b,eps=1e-3): return torch.sqrt((a-b)**2+eps**2).mean()

def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--data', default='data/clean_docs')
    p.add_argument('--paired-shadow'); p.add_argument('--paired-clean'); p.add_argument('--paired-mask')
    p.add_argument('--epochs',type=int,default=80); p.add_argument('--batch',type=int,default=16); p.add_argument('--size',type=int,default=1024)
    p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--base',type=int,default=48); p.add_argument('--workers',type=int,default=8)
    p.add_argument('--out',default='runs/doc_enhancer'); p.add_argument('--resume'); p.add_argument('--init-weights'); p.add_argument('--amp',action='store_true',default=True)
    p.add_argument('--grad-accum',type=int,default=1); p.add_argument('--device',default='cuda')
    return p.parse_args()

def main():
    a=parse(); seed_everything(42); out=Path(a.out); (out/'samples').mkdir(parents=True,exist_ok=True)
    if a.paired_shadow and a.paired_clean: ds=PairedShadowDataset(a.paired_shadow,a.paired_clean,a.paired_mask,a.size)
    else: ds=CleanDocSyntheticShadowDataset(a.data,a.size,augment=True)
    nval=max(1,int(len(ds)*0.05)); ntr=len(ds)-nval; tr,va=random_split(ds,[ntr,nval],generator=torch.Generator().manual_seed(42))
    dl=DataLoader(tr,batch_size=a.batch,shuffle=True,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    vl=DataLoader(va,batch_size=min(4,a.batch),shuffle=False,num_workers=max(1,a.workers//2),pin_memory=True)
    dev=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    if dev.type=='cuda' and getattr(torch.version,'hip',None):
        torch.backends.cudnn.enabled=False
    model=DocEnhancerNet(a.base).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    scaler=GradScaler(enabled=a.amp and dev.type=='cuda'); start=0; best=0
    if a.init_weights:
        ck=torch.load(a.init_weights,map_location='cpu',weights_only=False); model.load_state_dict(ck.get('model',ck))
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); start=ck.get('epoch',0)+1; best=ck.get('best',0)
    for e in range(start,a.epochs):
        model.train(); loss_sum=0; opt.zero_grad(set_to_none=True)
        for step,b in enumerate(tqdm(dl,desc=f'epoch {e+1}/{a.epochs}')):
            x=b['input'].to(dev,non_blocking=True); y=b['target'].to(dev,non_blocking=True); m=b['mask'].to(dev,non_blocking=True)
            with autocast(device_type='cuda', enabled=a.amp and dev.type=='cuda'):
                pred,pm=model(x)
            pred=pred.float(); pm=pm.float()
            loss=charbonnier(pred,y)+0.25*(1-torch.mean(torch.clamp(1-F.l1_loss(pred,y,reduction='none'),0,1)))+0.15*F.binary_cross_entropy(pm,m)+0.03*tv_loss(pred)
            loss=loss/a.grad_accum
            scaler.scale(loss).backward()
            if (step+1)%a.grad_accum==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())*a.grad_accum
        sched.step()
        model.eval(); scores=[]
        with torch.no_grad():
            for i,b in enumerate(vl):
                x=b['input'].to(dev); y=b['target'].to(dev); pred,pm=model(x)
                pn=(pred.clamp(0,1).cpu().numpy()*255).astype('uint8'); yn=(y.cpu().numpy()*255).astype('uint8')
                for k in range(pn.shape[0]):
                    pi=pn[k].transpose(1,2,0); yi=yn[k].transpose(1,2,0); scores.append(psnr(pi,yi))
                    if i==0 and k<2: imwrite_rgb(out/'samples'/f'e{e+1}_sample{k}.jpg',pi)
                if i>20: break
        val=sum(scores)/len(scores); isbest=val>best; best=max(best,val)
        ck={'model':model.state_dict(),'opt':opt.state_dict(),'epoch':e,'best':best,'args':vars(a)}
        torch.save(ck,out/'last.pth')
        if isbest: torch.save(ck,out/'best.pth')
        print(f'epoch={e+1} train_loss={loss_sum/max(1,len(dl)):.4f} val_psnr={val:.2f} best={best:.2f}')
if __name__=='__main__': main()
