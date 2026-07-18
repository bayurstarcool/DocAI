import argparse, torch
from pathlib import Path
from src.model import DocEnhancerNet
p=argparse.ArgumentParser(); p.add_argument('--weights',default='runs/doc_enhancer/best.pth'); p.add_argument('--out',default='exports/doc_enhancer.onnx'); p.add_argument('--size',type=int,default=1024); a=p.parse_args()
ck=torch.load(a.weights,map_location='cpu',weights_only=False)
base=ck.get('args',{}).get('base',48) if isinstance(ck,dict) else 48
model=DocEnhancerNet(base).eval(); model.load_state_dict(ck.get('model',ck))
x=torch.randn(1,3,a.size,a.size)
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
torch.onnx.export(model,x,a.out,input_names=['input'],output_names=['enhanced','shadow_mask'],opset_version=18,dynamic_axes={'input':{0:'batch',2:'height',3:'width'},'enhanced':{0:'batch',2:'height',3:'width'},'shadow_mask':{0:'batch',2:'height',3:'width'}},external_data=False)
print('saved',a.out)
