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


def apply_color_temperature_shift(image, mask):
    """Apply warm/cool color shift to shadow areas (realistic lighting)."""
    img = image.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    if random.random() < 0.7:
        temp = np.array([random.uniform(1.02, 1.12), random.uniform(0.94, 1.0), random.uniform(0.88, 0.96)], dtype=np.float32)
    else:
        temp = np.array([random.uniform(0.92, 0.98), random.uniform(0.96, 1.02), random.uniform(1.02, 1.12)], dtype=np.float32)
    shadow_mask = mask.astype(np.float32) / 255.0
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), 3.0)
    shadow_mask = np.clip(shadow_mask * random.uniform(0.3, 0.7), 0, 1)
    result = img.copy()
    for c in range(3):
        result[..., c] = img[..., c] * (1 - shadow_mask * (1 - temp[c]))
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def apply_ambient_occlusion(image, mask):
    """Add edge darkening at shadow boundaries."""
    img = image.astype(np.float32) / 255.0
    shadow_mask = mask.astype(np.float32) / 255.0
    edges = cv2.Canny((shadow_mask * 255).astype(np.uint8), 30, 100)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 5.0)
    edges = edges / (edges.max() + 1e-6)
    occlusion_strength = random.uniform(0.15, 0.35)
    result = img * (1 - edges[..., None] * occlusion_strength)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def apply_bounce_light(image, mask):
    """Add subtle bounce light in shadow areas."""
    img = image.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    shadow_mask = mask.astype(np.float32) / 255.0
    gradient = np.zeros((h, w), dtype=np.float32)
    side = random.choice(["left", "right", "top", "bottom"])
    if side == "left":
        gradient = np.linspace(0.8, 1.0, w)[None, :]
    elif side == "right":
        gradient = np.linspace(1.0, 0.8, w)[None, :]
    elif side == "top":
        gradient = np.linspace(0.8, 1.0, h)[:, None]
    else:
        gradient = np.linspace(1.0, 0.8, h)[:, None]
    gradient = cv2.GaussianBlur(gradient, (0, 0), max(h // 4, 1))
    bounce_strength = random.uniform(0.05, 0.15)
    # Bounce light must brighten shadow pixels. Previous `(gradient - 1)`
    # term was negative and therefore darkened them further.
    result = img * (1 + shadow_mask[..., None] * gradient[..., None] * bounce_strength)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def apply_realistic_shadow(image, return_mask=False):
    """Apply realistic shadow with color temperature, ambient occlusion, and bounce light."""
    img = image.copy()
    h, w = img.shape[:2]
    if random.random() < 0.6:
        shadow_img, mask = apply_object_shadow(img, return_mask=True)
    else:
        shadow_img, mask = apply_document_shadow(img, return_mask=True)
    if random.random() < 0.75:
        shadow_img = apply_color_temperature_shift(shadow_img, mask)
    if random.random() < 0.5:
        shadow_img = apply_ambient_occlusion(shadow_img, mask)
    if random.random() < 0.4:
        shadow_img = apply_bounce_light(shadow_img, mask)
    if return_mask:
        return shadow_img, mask
    return shadow_img
