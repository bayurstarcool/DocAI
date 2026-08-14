"""SAM document crop with auto-rotation for CamScanner-like output.
Runs on CPU to avoid OOM with DocAI GPU cache."""
import sys, cv2, numpy as np, torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

inp=Path(sys.argv[1]); out=Path(sys.argv[2])
img_bgr=cv2.imread(str(inp))
if img_bgr is None: raise SystemExit('input image not readable')
image=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H,W=image.shape[:2]

# Force CPU to avoid GPU OOM with DocAI server
sam=sam_model_registry['vit_b'](checkpoint='/home/wahyu/models/sam/sam_vit_b_01ec64.pth')
sam.to('cpu')
sam.eval()

def approx_doc_from_mask(mask):
    mask=(mask.astype(np.uint8)*255)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c=max(cnts,key=cv2.contourArea)
    peri=cv2.arcLength(c,True)
    for eps in [0.01,0.015,0.02,0.03,0.04,0.06]:
        approx=cv2.approxPolyDP(c,eps*peri,True)
        if len(approx)==4:
            return approx.reshape(4,2).astype(np.float32)
    rect=cv2.minAreaRect(c)
    return cv2.boxPoints(rect).astype(np.float32)

def order_points(pts):
    rect=np.zeros((4,2),dtype=np.float32)
    s=pts.sum(axis=1); rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
    d=np.diff(pts,axis=1); rect[1]=pts[np.argmin(d)]; rect[3]=pts[np.argmax(d)]
    return rect

def warp(img,pts):
    rect=order_points(pts); tl,tr,br,bl=rect
    w=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl)))
    h=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
    w=max(w,64); h=max(h,64)
    dst=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype=np.float32)
    M=cv2.getPerspectiveTransform(rect,dst)
    return cv2.warpPerspective(img,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def auto_rotate_by_lines(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,50,150,apertureSize=3)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=80,minLineLength=min(img.shape[:2])//6,maxLineGap=20)
    if lines is None: return img
    angles=[]
    for l in lines:
        coords = l.reshape(-1)
        if len(coords)==4:
            x1,y1,x2,y2=coords
            a=np.degrees(np.arctan2(y2-y1,x2-x1))
            if abs(a)<30: angles.append(a)
    if not angles: return img
    median_angle=np.median(angles)
    if abs(median_angle)<0.3: return img
    center=(img.shape[1]//2, img.shape[0]//2)
    M=cv2.getRotationMatrix2D(center,median_angle,1.0)
    return cv2.warpAffine(img,M,(img.shape[1],img.shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def auto_orient_portrait(img):
    h,w=img.shape[:2]
    if w > h:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

predictor=SamPredictor(sam)
predictor.set_image(image)
box=np.array([3,max(0,int(0.02*H)),W-3,H-3])
points=np.array([[W//2,H//2],[20,20],[W-20,20]])
labels=np.array([1,0,0])
masks,scores,_=predictor.predict(point_coords=points,point_labels=labels,box=box,multimask_output=True)
best=masks[int(np.argmax(scores))]
pts=approx_doc_from_mask(best)

if pts is None or (best.sum()/(W*H)<0.20):
    mask_gen=SamAutomaticMaskGenerator(sam,points_per_side=16,pred_iou_thresh=0.85,stability_score_thresh=0.90,min_mask_region_area=5000)
    auto=mask_gen.generate(image)
    cand=[m for m in auto if 0.20<m['area']/(W*H)<0.98]
    cand=sorted(cand,key=lambda m:(m['predicted_iou']+m['stability_score'],m['area']),reverse=True)
    if cand: pts=approx_doc_from_mask(cand[0]['segmentation'])

if pts is None:
    result=img_bgr
else:
    result=warp(img_bgr,pts)

result=auto_orient_portrait(result)
result=auto_rotate_by_lines(result)
cv2.imwrite(str(out),result)
print('ok',result.shape)
