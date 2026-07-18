from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6): super().__init__(); self.w=nn.Parameter(torch.ones(c)); self.b=nn.Parameter(torch.zeros(c)); self.eps=eps
    def forward(self,x):
        u=x.mean(1,keepdim=True); s=(x-u).pow(2).mean(1,keepdim=True)
        return (x-u)/torch.sqrt(s+self.eps)*self.w[:,None,None]+self.b[:,None,None]

class GatedDWBlock(nn.Module):
    def __init__(self,c,exp=2):
        super().__init__(); h=c*exp
        self.n1=LayerNorm2d(c); self.pw1=nn.Conv2d(c,h*2,1); self.dw=nn.Conv2d(h*2,h*2,3,1,1,groups=h*2); self.pw2=nn.Conv2d(h,c,1)
        self.n2=LayerNorm2d(c); self.ff=nn.Sequential(nn.Conv2d(c,h,1),nn.GELU(),nn.Conv2d(h,c,1))
    def forward(self,x):
        y=self.dw(self.pw1(self.n1(x))); a,b=y.chunk(2,1); x=x+self.pw2(a*torch.sigmoid(b)); return x+self.ff(self.n2(x))

class Down(nn.Module):
    def __init__(self,cin,cout): super().__init__(); self.net=nn.Sequential(nn.Conv2d(cin,cout,3,2,1),GatedDWBlock(cout),GatedDWBlock(cout))
    def forward(self,x): return self.net(x)
class Up(nn.Module):
    def __init__(self,cin,skip,cout): super().__init__(); self.up=nn.ConvTranspose2d(cin,cout,2,2); self.fuse=nn.Sequential(nn.Conv2d(cout+skip,cout,1),GatedDWBlock(cout),GatedDWBlock(cout))
    def forward(self,x,s):
        x=self.up(x)
        if x.shape[-2:]!=s.shape[-2:]: x=F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False)
        return self.fuse(torch.cat([x,s],1))

class DocEnhancerNet(nn.Module):
    """Production-friendly document shadow remover.
    Outputs enhanced RGB and shadow mask. Works with ONNX/TensorRT.
    """
    def __init__(self, base=48):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,base,3,1,1),GatedDWBlock(base),GatedDWBlock(base))
        self.d1=Down(base,base*2); self.d2=Down(base*2,base*4); self.d3=Down(base*4,base*8)
        self.mid=nn.Sequential(*[GatedDWBlock(base*8) for _ in range(6)])
        self.u3=Up(base*8,base*4,base*4); self.u2=Up(base*4,base*2,base*2); self.u1=Up(base*2,base,base)
        self.rgb=nn.Conv2d(base,3,3,1,1); self.mask=nn.Conv2d(base,1,3,1,1)
    def forward(self,x):
        s0=self.stem(x); s1=self.d1(s0); s2=self.d2(s1); z=self.mid(self.d3(s2))
        y=self.u3(z,s2); y=self.u2(y,s1); y=self.u1(y,s0)
        residual=torch.tanh(self.rgb(y))*0.35
        enhanced=torch.clamp(x+residual,0,1)
        mask=torch.sigmoid(self.mask(y))
        return enhanced, mask
