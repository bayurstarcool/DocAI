from __future__ import annotations
import cv2, numpy as np, random, math

def _smooth_noise(h,w,scale=6):
    small = np.random.rand(max(2,h//scale), max(2,w//scale)).astype(np.float32)
    noise = cv2.resize(small,(w,h),interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise,(0,0),sigmaX=max(h,w)/35)
    return (noise-noise.min())/(noise.max()-noise.min()+1e-6)

def _clean_shadow_mask(mask):
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)
    mask = np.clip((mask - 0.04) / 0.42, 0, 1)
    return mask.astype(np.float32)

def random_shadow_mask(h,w):
    mask = np.zeros((h,w), np.float32)
    # soft blobs / hand-like shadows
    for _ in range(random.randint(2,6)):
        cx,cy=random.randint(-w//5,w+w//5), random.randint(-h//5,h+h//5)
        ax,ay=random.randint(w//8,w//2), random.randint(h//10,h//2)
        angle=random.randint(0,180)
        tmp=np.zeros_like(mask)
        cv2.ellipse(tmp,(cx,cy),(ax,ay),angle,0,360,1,-1)
        blur=random.randint(81,251)|1
        tmp=cv2.GaussianBlur(tmp,(blur,blur),0)
        mask=np.maximum(mask,tmp*random.uniform(.35,1.0))
    # strip/edge shadow
    if random.random()<.7:
        x1=random.randint(-w//3,w); x2=x1+random.randint(w//8,w//2)
        poly=np.array([[x1,0],[x2,0],[x2+random.randint(-w//3,w//3),h],[x1+random.randint(-w//3,w//3),h]],np.int32)
        tmp=np.zeros_like(mask); cv2.fillPoly(tmp,[poly],1)
        tmp=cv2.GaussianBlur(tmp,(151,151),0)
        mask=np.maximum(mask,tmp*random.uniform(.25,.85))
    mask *= _smooth_noise(h,w,random.randint(4,12))*0.7+0.3
    return _clean_shadow_mask(mask)

def _line_shadow(h,w,angle=None,width=None,offset=None,blur=None,strength=None):
    angle = random.uniform(-35,35) if angle is None else angle
    width = random.uniform(w*.035,w*.11) if width is None else width
    offset = random.uniform(-w*.35,w*.35) if offset is None else offset
    strength = random.uniform(.35,.9) if strength is None else strength
    yy,xx=np.mgrid[0:h,0:w]
    cx=w*.5+offset; cy=h*.5
    theta=math.radians(angle)
    dist=(xx-cx)*math.cos(theta)+(yy-cy)*math.sin(theta)
    mask=np.exp(-(dist**2)/(2*width**2)).astype(np.float32)*strength
    blur = int(max(31, (blur if blur is not None else random.uniform(w*.025,w*.08))))|1
    return cv2.GaussianBlur(mask,(blur,blur),0)

def _finger_mask(h,w):
    mask=np.zeros((h,w),np.float32)
    count=random.randint(2,5)
    base_angle=random.choice([random.uniform(-18,12), random.uniform(72,108)])
    spacing=random.uniform(w*.055,w*.12)
    center=random.uniform(-w*.12,w*.18)
    for i in range(count):
        offset=center+(i-(count-1)/2)*spacing
        width=random.uniform(w*.025,w*.06)
        line=_line_shadow(h,w,base_angle+random.uniform(-7,7),width,offset,random.uniform(w*.025,w*.065),random.uniform(.38,.85))
        taper=np.linspace(random.uniform(.5,1.0),random.uniform(.35,1.0),h,dtype=np.float32)[:,None]
        mask=np.maximum(mask,line*taper)
    return np.clip(mask,0,1)

def _hand_blob_mask(h,w):
    mask=np.zeros((h,w),np.float32)
    side=random.choices(['bottom','left','right','top'], weights=[5,2,2,1], k=1)[0]
    if side=='bottom': cx=random.randint(0,w); cy=random.randint(int(h*.82),int(h*1.12))
    elif side=='top': cx=random.randint(0,w); cy=random.randint(-int(h*.12),int(h*.18))
    elif side=='left': cx=random.randint(-int(w*.12),int(w*.18)); cy=random.randint(0,h)
    else: cx=random.randint(int(w*.82),int(w*1.12)); cy=random.randint(0,h)
    for _ in range(random.randint(2,4)):
        tmp=np.zeros_like(mask)
        ax=random.randint(max(10,w//9),max(20,w//3)); ay=random.randint(max(10,h//10),max(20,h//3))
        ox=random.randint(-w//8,w//8); oy=random.randint(-h//8,h//8)
        cv2.ellipse(tmp,(cx+ox,cy+oy),(ax,ay),random.randint(0,180),0,360,1,-1)
        blur=random.randint(61,191)|1
        tmp=cv2.GaussianBlur(tmp,(blur,blur),0)
        mask=np.maximum(mask,tmp*random.uniform(.35,.9))
    return np.clip(mask,0,1)

def _edge_cast_mask(h,w):
    yy,xx=np.mgrid[0:h,0:w]
    mask=np.zeros((h,w),np.float32)
    side=random.choice(['bottom','left','right','top'])
    if side=='bottom': dist=h-yy
    elif side=='top': dist=yy
    elif side=='left': dist=xx
    else: dist=w-xx
    width=random.uniform(min(h,w)*.08,min(h,w)*.25)
    mask=np.clip(1-dist/width,0,1)**2
    mask=cv2.GaussianBlur(mask.astype(np.float32),(random.randint(81,251)|1,random.randint(81,251)|1),0)
    return mask*random.uniform(.25,.7)

def object_shadow_mask(h,w):
    mask=np.zeros((h,w),np.float32)
    modes=random.sample(['fingers','hand','edge'], k=random.randint(1,3))
    if random.random()<.75 and 'hand' not in modes:
        modes.append('hand')
    if 'fingers' in modes: mask=np.maximum(mask,_finger_mask(h,w))
    if 'hand' in modes: mask=np.maximum(mask,_hand_blob_mask(h,w))
    if 'edge' in modes: mask=np.maximum(mask,_edge_cast_mask(h,w))
    noise=_smooth_noise(h,w,random.randint(5,14))*0.35+0.75
    return _clean_shadow_mask(np.clip(mask*noise,0,1))

def apply_object_shadow(clean_rgb, return_mask=False):
    img=clean_rgb.astype(np.float32)/255.0
    h,w=img.shape[:2]
    mask=object_shadow_mask(h,w)
    strength=random.uniform(.58,.94)
    warm=np.array([random.uniform(.50,.78), random.uniform(.48,.74), random.uniform(.44,.70)],np.float32)
    shaded=img*(1-mask[...,None]*strength*warm)
    vignette=random_shadow_mask(h,w)*random.uniform(.08,.28)
    shaded=shaded*(1-vignette[...,None])
    if random.random()<.65:
        shaded=np.clip(shaded+np.random.normal(0,random.uniform(.002,.01),shaded.shape),0,1)
    out=(shaded*255).astype(np.uint8)
    sm=(np.clip(mask+vignette*.45,0,1)*255).astype(np.uint8)
    return (out, sm) if return_mask else out

def apply_document_shadow(clean_rgb, return_mask=False):
    img=clean_rgb.astype(np.float32)/255.0
    h,w=img.shape[:2]
    mask=random_shadow_mask(h,w)
    strength=random.uniform(.35,.78)
    color=np.array([random.uniform(.55,.85), random.uniform(.55,.85), random.uniform(.55,.85)],np.float32)
    shaded=img*(1-mask[...,None]*strength*color)
    # uneven illumination and camera noise
    yy,xx=np.mgrid[0:h,0:w]
    grad=((xx/w)*random.uniform(-.16,.16)+(yy/h)*random.uniform(-.16,.16)+1.0)
    shaded=np.clip(shaded*grad[...,None],0,1)
    if random.random()<.7:
        shaded=np.clip(shaded+np.random.normal(0,random.uniform(.003,.015),shaded.shape),0,1)
    out=(shaded*255).astype(np.uint8)
    sm=(mask*255).astype(np.uint8)
    return (out, sm) if return_mask else out
