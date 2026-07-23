"""
Dataset Audit & Preparation Utilities
- Audit image pairs for feasibility as training data
- Auto-fix: resize, align, register
- Validate dataset quality
"""
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
import json
import shutil
import time


def compute_luma(img: np.ndarray) -> np.ndarray:
    """Compute luminance channel."""
    return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]


def compute_sharpness(gray: np.ndarray) -> float:
    """Compute image sharpness via Laplacian variance."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_shadow_ratio(luma: np.ndarray, threshold: int = 128) -> float:
    """Compute percentage of dark pixels."""
    return float((luma < threshold).sum() / luma.size * 100)


def align_images_orb(ref: np.ndarray, target: np.ndarray, max_features: int = 1000) -> Tuple[np.ndarray, Dict]:
    """
    Align target image to reference using ORB feature matching + homography.
    Returns aligned target image and alignment metadata.
    """
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(tgt_gray, None)

    meta = {
        "matches_total": 0,
        "matches_good": 0,
        "inliers": 0,
        "tx": 0.0,
        "ty": 0.0,
        "scale": 1.0,
        "success": False,
        "method": "orb_homography"
    }

    if des1 is None or des2 is None:
        return target, meta

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = [m for m in matches if m.distance < 50]

    meta["matches_total"] = len(matches)
    meta["matches_good"] = len(good)

    if len(good) < 10:
        return target, meta

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if M is None:
        return target, meta

    inliers = int(mask.ravel().sum()) if mask is not None else 0
    meta["inliers"] = inliers
    meta["tx"] = float(M[0, 2])
    meta["ty"] = float(M[1, 2])
    meta["scale"] = float(np.sqrt(M[0,0]**2 + M[1,0]**2))

    h, w = ref.shape[:2]
    aligned = cv2.warpPerspective(target, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    meta["success"] = True

    return aligned, meta


def resize_to_match(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize image to match target (height, width)."""
    h, w = target_shape[:2]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)


def audit_pair(shadow_path: str, clean_path: str) -> Dict:
    """
    Audit a shadow/clean image pair for dataset feasibility.
    Returns detailed audit report.
    """
    shadow_img = np.array(Image.open(shadow_path).convert("RGB"))
    clean_img = np.array(Image.open(clean_path).convert("RGB"))

    shadow_h, shadow_w = shadow_img.shape[:2]
    clean_h, clean_w = clean_img.shape[:2]

    # Luminance
    luma_shadow = compute_luma(shadow_img)
    luma_clean = compute_luma(clean_img)

    # Shadow ratio
    shadow_pct = compute_shadow_ratio(luma_shadow)
    clean_dark_pct = compute_shadow_ratio(luma_clean)

    # Sharpness
    sharp_shadow = compute_sharpness(cv2.cvtColor(shadow_img, cv2.COLOR_RGB2GRAY))
    sharp_clean = compute_sharpness(cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY))

    # Color channels
    color_diff = {}
    for name, ch in [("R", 0), ("G", 1), ("B", 2)]:
        c_mean = float(clean_img[:,:,ch].mean())
        s_mean = float(shadow_img[:,:,ch].mean())
        color_diff[name] = {"clean": round(c_mean, 1), "shadow": round(s_mean, 1), "diff": round(s_mean - c_mean, 1)}

    # Alignment check
    # Resize shadow to clean size for alignment check
    if (shadow_h, shadow_w) != (clean_h, clean_w):
        shadow_resized = resize_to_match(shadow_img, clean_img.shape)
    else:
        shadow_resized = shadow_img

    aligned, align_meta = align_images_orb(clean_img, shadow_resized)

    # Difference map
    luma_aligned = compute_luma(aligned)
    diff_map = np.abs(luma_clean.astype(float) - luma_aligned.astype(float))
    mean_diff = float(diff_map.mean())
    high_diff_pct = float((diff_map > 30).sum() / diff_map.size * 100)

    # Text preservation check
    gray_clean = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
    text_mask = (cv2.Laplacian(gray_clean, cv2.CV_64F) > 20).astype(float)
    if text_mask.sum() > 0:
        text_low_diff = float((text_mask * (diff_map < 20)).sum() / text_mask.sum() * 100)
    else:
        text_low_diff = 100.0

    # Score calculation
    score = 0
    reasons = []

    # Shadow must be present (shadow_pct > 5%)
    if shadow_pct > 5:
        score += 25
    else:
        reasons.append("Shadow terlalu sedikit (<5%)")

    # Clean should be brighter than shadow
    if luma_clean.mean() > luma_shadow.mean() + 10:
        score += 20
    else:
        reasons.append("Clean tidak cukup terang dari shadow")

    # Shadow image should have more dark pixels
    if shadow_pct > clean_dark_pct + 3:
        score += 15
    else:
        reasons.append("Shadow tidak cukup berbeda dari clean")

    # Alignment should be good
    if align_meta["success"] and align_meta["inliers"] > 50:
        score += 20
    elif align_meta["success"]:
        score += 10
        reasons.append("Alignment kurang presisi")
    else:
        reasons.append("Alignment gagal")

    # Resolution difference
    res_diff = abs(shadow_w * shadow_h - clean_w * clean_h) / (clean_w * clean_h) * 100
    if res_diff < 10:
        score += 10
    elif res_diff < 50:
        score += 5
        reasons.append(f"Resolusi beda {res_diff:.0f}%")
    else:
        reasons.append(f"Resolusi beda terlalu besar ({res_diff:.0f}%)")

    # Color preservation
    if abs(color_diff["R"]["diff"]) > 10 or abs(color_diff["G"]["diff"]) > 10:
        score += 10
    else:
        reasons.append("Color difference terlalu kecil")

    # Grade
    if score >= 80:
        grade = "A"
    elif score >= 60:
        grade = "B"
    elif score >= 40:
        grade = "C"
    else:
        grade = "D"

    return {
        "score": score,
        "grade": grade,
        "reasons": reasons,
        "dimensions": {
            "shadow": {"w": shadow_w, "h": shadow_h},
            "clean": {"w": clean_w, "h": clean_h},
            "match": (shadow_h, shadow_w) == (clean_h, clean_w),
            "resolution_diff_pct": round(res_diff, 1)
        },
        "luminance": {
            "shadow_mean": round(float(luma_shadow.mean()), 1),
            "clean_mean": round(float(luma_clean.mean()), 1),
            "shadow_pct": round(shadow_pct, 1),
            "clean_dark_pct": round(clean_dark_pct, 1)
        },
        "color_diff": color_diff,
        "sharpness": {
            "shadow": round(sharp_shadow, 1),
            "clean": round(sharp_clean, 1)
        },
        "alignment": align_meta,
        "diff_map": {
            "mean": round(mean_diff, 1),
            "high_diff_pct": round(high_diff_pct, 1),
            "text_preserved_pct": round(text_low_diff, 1)
        }
    }


def get_next_pair_number(output_dir: str) -> int:
    """Get next available pair number for sequential naming."""
    input_dir = os.path.join(output_dir, "input")
    if not os.path.exists(input_dir):
        return 1
    existing = [f for f in os.listdir(input_dir) if f.startswith("pair_") and f.endswith(".png")]
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.replace("pair_", "").replace(".png", ""))
            numbers.append(num)
        except ValueError:
            pass
    return max(numbers) + 1 if numbers else 1


def prepare_pair(shadow_path: str, clean_path: str, output_dir: str,
                 target_size: Tuple[int, int] = (768, 768),
                 align: bool = True) -> Dict:
    """
    Prepare a shadow/clean pair for training:
    1. Resize both to target_size
    2. Align shadow to clean
    3. Save to output_dir/input/ and output_dir/target/
    4. Auto-increment numbering (pair_0001, pair_0002, ...)
    """
    shadow_img = np.array(Image.open(shadow_path).convert("RGB"))
    clean_img = np.array(Image.open(clean_path).convert("RGB"))

    os.makedirs(os.path.join(output_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "target"), exist_ok=True)

    # Get next sequential number
    pair_num = get_next_pair_number(output_dir)
    pair_name = f"pair_{pair_num:04d}"

    # Resize maintaining aspect ratio, max dimension = max_size
    def resize_max(img, max_size=768):
        ih, iw = img.shape[:2]
        if max(ih, iw) <= max_size:
            return img
        scale = max_size / max(ih, iw)
        new_w = int(iw * scale)
        new_h = int(ih * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    clean_resized = resize_max(clean_img, 768)
    shadow_resized = resize_max(shadow_img, 768)

    # Align shadow to clean if requested
    align_meta = {}
    if align:
        shadow_resized, align_meta = align_images_orb(clean_resized, shadow_resized)

    # Save with sequential name
    input_path = os.path.join(output_dir, "input", f"{pair_name}.png")
    target_path = os.path.join(output_dir, "target", f"{pair_name}.png")

    Image.fromarray(shadow_resized).save(input_path)
    Image.fromarray(clean_resized).save(target_path)

    return {
        "input": input_path,
        "target": target_path,
        "pair_name": pair_name,
        "pair_number": pair_num,
        "size": target_size,
        "alignment": align_meta
    }


def prepare_dataset(shadow_dir: str, clean_dir: str, output_dir: str,
                    target_size: Tuple[int, int] = (768, 768),
                    align: bool = True) -> Dict:
    """
    Prepare an entire dataset of shadow/clean pairs.
    Expects shadow_dir and clean_dir to have images with matching filenames.
    """
    shadow_files = sorted([f for f in os.listdir(shadow_dir)
                          if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))])
    clean_files = sorted([f for f in os.listdir(clean_dir)
                         if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))])

    # Match files by name (without extension)
    shadow_names = {Path(f).stem: f for f in shadow_files}
    clean_names = {Path(f).stem: f for f in clean_files}

    matched = set(shadow_names.keys()) & set(clean_names.keys())
    unmatched_shadow = set(shadow_names.keys()) - matched
    unmatched_clean = set(clean_names.keys()) - matched

    results = {
        "total_pairs": len(matched),
        "processed": 0,
        "errors": [],
        "pairs": []
    }

    for name in sorted(matched):
        shadow_path = os.path.join(shadow_dir, shadow_names[name])
        clean_path = os.path.join(clean_dir, clean_names[name])

        try:
            pair_result = prepare_pair(shadow_path, clean_path, output_dir, target_size, align)
            results["pairs"].append({"name": name, **pair_result})
            results["processed"] += 1
        except Exception as e:
            results["errors"].append({"name": name, "error": str(e)})

    results["unmatched_shadow"] = list(unmatched_shadow)
    results["unmatched_clean"] = list(unmatched_clean)

    return results


def validate_dataset(dataset_dir: str) -> Dict:
    """
    Validate all pairs in a prepared dataset directory.
    Expects dataset_dir/input/ and dataset_dir/target/ with matching filenames.
    """
    input_dir = os.path.join(dataset_dir, "input")
    target_dir = os.path.join(dataset_dir, "target")

    if not os.path.exists(input_dir) or not os.path.exists(target_dir):
        return {"valid": False, "error": "Dataset must have input/ and target/ directories"}

    input_files = {Path(f).stem: f for f in os.listdir(input_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))}
    target_files = {Path(f).stem: f for f in os.listdir(target_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))}

    matched = set(input_files.keys()) & set(target_files.keys())
    pairs = []

    for name in sorted(matched):
        input_path = os.path.join(input_dir, input_files[name])
        target_path = os.path.join(target_dir, target_files[name])

        try:
            audit = audit_pair(input_path, target_path)
            pairs.append({"name": name, "audit": audit})
        except Exception as e:
            pairs.append({"name": name, "error": str(e)})

    # Summary
    grades = [p["audit"]["grade"] for p in pairs if "audit" in p]
    avg_score = np.mean([p["audit"]["score"] for p in pairs if "audit" in p]) if pairs else 0

    return {
        "valid": len(pairs) > 0,
        "total_pairs": len(pairs),
        "grades": {g: grades.count(g) for g in ["A", "B", "C", "D"]},
        "average_score": round(float(avg_score), 1),
        "pairs": pairs
    }
# Backend: dataset CRUD helpers
# Add to backend/utils/dataset_audit.py

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

DATASETS_DIR = "datasets/paired/custom"


def get_datasets_dir() -> str:
    return DATASETS_DIR


def list_datasets() -> List[Dict]:
    """List all custom datasets with metadata."""
    base = DATASETS_DIR
    if not os.path.exists(base):
        os.makedirs(base, exist_ok=True)
        return []

    datasets = []
    for slug in sorted(os.listdir(base)):
        ds_path = os.path.join(base, slug)
        if not os.path.isdir(ds_path):
            continue
        meta = _load_meta(ds_path)
        input_dir = os.path.join(ds_path, "input")
        target_dir = os.path.join(ds_path, "target")
        input_count = len([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(input_dir) else 0
        target_count = len([f for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(target_dir) else 0

        datasets.append({
            "slug": slug,
            "name": meta.get("name", slug),
            "description": meta.get("description", ""),
            "input_count": input_count,
            "target_count": target_count,
            "paired_count": min(input_count, target_count),
            "created_at": meta.get("created_at", ""),
            "target_size": meta.get("target_size", "768,768"),
            "status": "ready" if min(input_count, target_count) > 0 else "empty",
        })

    return datasets


def get_dataset(slug: str) -> Optional[Dict]:
    """Get dataset info by slug."""
    ds_path = os.path.join(DATASETS_DIR, slug)
    if not os.path.exists(ds_path) or not os.path.isdir(ds_path):
        return None

    meta = _load_meta(ds_path)
    input_dir = os.path.join(ds_path, "input")
    target_dir = os.path.join(ds_path, "target")

    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(input_dir) else []
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(target_dir) else []

    pairs = []
    for inp in input_files:
        stem = Path(inp).stem
        tgt = f"{stem}.png"
        has_target = tgt in target_files
        pairs.append({
            "name": stem,
            "input": f"input/{inp}",
            "target": f"target/{tgt}" if has_target else None,
            "paired": has_target,
        })

    return {
        "slug": slug,
        "name": meta.get("name", slug),
        "description": meta.get("description", ""),
        "created_at": meta.get("created_at", ""),
        "target_size": meta.get("target_size", "768,768"),
        "input_count": len(input_files),
        "target_count": len(target_files),
        "paired_count": sum(1 for p in pairs if p["paired"]),
        "pairs": pairs,
    }


def create_dataset(slug: str, name: str = "", description: str = "", target_size: str = "768,768") -> Dict:
    """Create a new empty dataset."""
    # Sanitize slug
    slug = slug.lower().strip().replace(" ", "-").replace("_", "-")
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    if not slug:
        return {"error": "Invalid slug"}

    ds_path = os.path.join(DATASETS_DIR, slug)
    if os.path.exists(ds_path):
        return {"error": f"Dataset '{slug}' already exists"}

    os.makedirs(os.path.join(ds_path, "input"), exist_ok=True)
    os.makedirs(os.path.join(ds_path, "target"), exist_ok=True)

    import time
    meta = {
        "name": name or slug,
        "description": description,
        "slug": slug,
        "target_size": target_size,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_meta(ds_path, meta)

    return {"slug": slug, "name": meta["name"], "created_at": meta["created_at"]}


def delete_dataset(slug: str) -> bool:
    """Delete a dataset by slug."""
    ds_path = os.path.join(DATASETS_DIR, slug)
    if not os.path.exists(ds_path):
        return False
    shutil.rmtree(ds_path)
    return True


def update_dataset_meta(slug: str, name: str = None, description: str = None) -> Dict:
    """Update dataset metadata."""
    ds_path = os.path.join(DATASETS_DIR, slug)
    if not os.path.exists(ds_path):
        return {"error": "Dataset not found"}

    meta = _load_meta(ds_path)
    if name is not None:
        meta["name"] = name
    if description is not None:
        meta["description"] = description
    _save_meta(ds_path, meta)
    return {"slug": slug, **meta}


def _meta_path(ds_path: str) -> str:
    return os.path.join(ds_path, "dataset.json")


def _load_meta(ds_path: str) -> Dict:
    path = _meta_path(ds_path)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_meta(ds_path: str, meta: Dict):
    with open(_meta_path(ds_path), "w") as f:
        json.dump(meta, f, indent=2)
"""
Dataset Quality Validation
- Validates all pairs in a dataset
- Checks: sharpness, alignment, shadow presence, color, resolution
- Returns per-pair score + overall dataset quality
"""
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List


def compute_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_shadow_ratio(luma: np.ndarray, threshold: int = 128) -> float:
    return float((luma < threshold).sum() / luma.size * 100)


def check_alignment(ref: np.ndarray, target: np.ndarray) -> Dict:
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(tgt_gray, None)

    result = {"success": False, "inliers": 0, "tx": 0.0, "ty": 0.0, "score": 0}

    if des1 is None or des2 is None:
        return result

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = [m for m in matches if m.distance < 50]

    if len(good) < 10:
        return result

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if M is None:
        return result

    inliers = int(mask.ravel().sum()) if mask is not None else 0
    tx = float(M[0, 2])
    ty = float(M[1, 2])

    result["success"] = True
    result["inliers"] = inliers
    result["tx"] = round(tx, 1)
    result["ty"] = round(ty, 1)
    result["score"] = min(100, int(inliers / len(good) * 100)) if good else 0

    return result


def validate_pair(input_path: str, target_path: str) -> Dict:
    """Validate a single pair. Returns score 0-100 + issues."""
    issues = []
    score = 0

    try:
        input_img = np.array(Image.open(input_path).convert("RGB"))
        target_img = np.array(Image.open(target_path).convert("RGB"))
    except Exception as e:
        return {"score": 0, "grade": "F", "issues": [f"Cannot load: {e}"]}

    ih, iw = input_img.shape[:2]
    th, tw = target_img.shape[:2]

    # 1. Resolution check (20 pts)
    min_dim = min(ih, iw, th, tw)
    if min_dim >= 720:
        score += 20
    elif min_dim >= 480:
        score += 15
    elif min_dim >= 320:
        score += 10
    else:
        issues.append(f"Resolusi terlalu rendah ({min_dim}px)")

    # 2. Sharpness check - input (15 pts)
    input_sharp = compute_sharpness(cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY))
    if input_sharp > 200:
        score += 15
    elif input_sharp > 50:
        score += 10
    else:
        issues.append(f"Input blur (sharpness={input_sharp:.0f})")

    # 3. Sharpness check - target (15 pts)
    target_sharp = compute_sharpness(cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY))
    if target_sharp > 200:
        score += 15
    elif target_sharp > 50:
        score += 10
    else:
        issues.append(f"Target blur (sharpness={target_sharp:.0f})")

    # 4. Shadow presence in input (15 pts)
    input_luma = 0.299 * input_img[:,:,0] + 0.587 * input_img[:,:,1] + 0.114 * input_img[:,:,2]
    shadow_pct = compute_shadow_ratio(input_luma)
    if shadow_pct >= 5:
        score += 15
    elif shadow_pct >= 2:
        score += 10
    else:
        issues.append(f"Shadow terlalu sedikit ({shadow_pct:.1f}%)")

    # 5. Target should be brighter than input (10 pts)
    target_luma = 0.299 * target_img[:,:,0] + 0.587 * target_img[:,:,1] + 0.114 * target_img[:,:,2]
    if target_luma.mean() > input_luma.mean() + 5:
        score += 10
    else:
        issues.append("Target tidak lebih terang dari input")

    # 6. Alignment check (15 pts)
    if (ih, iw) == (th, tw):
        align = check_alignment(target_img, input_img)
        if align["success"] and align["inliers"] > 30:
            score += 15
            if align["tx"] > 10 or align["ty"] > 10:
                issues.append(f"Alignment longgar (tx={align['tx']}px, ty={align['ty']}px)")
        elif align["success"]:
            score += 8
            issues.append(f"Alignment kurang presisi ({align['inliers']} inliers)")
        else:
            issues.append("Alignment gagal")
    else:
        issues.append(f"Ukuran beda: input={iw}x{ih}, target={tw}x{th}")

    # 7. Color diversity check (10 pts)
    color_std = input_img.std()
    if color_std > 40:
        score += 10
    elif color_std > 20:
        score += 5
    else:
        issues.append("Input kurang variasi warna")

    # Grade
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 50:
        grade = "C"
    elif score >= 30:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": score,
        "grade": grade,
        "issues": issues,
        "input_size": f"{iw}x{ih}",
        "target_size": f"{tw}x{th}",
        "input_sharpness": round(input_sharp, 1),
        "target_sharpness": round(target_sharp, 1),
        "shadow_ratio": round(shadow_pct, 1),
        "alignment": align if (ih, iw) == (th, tw) else None,
    }


def validate_dataset(dataset_dir: str) -> Dict:
    """Validate all pairs in a dataset directory."""
    input_dir = os.path.join(dataset_dir, "input")
    target_dir = os.path.join(dataset_dir, "target")

    if not os.path.exists(input_dir) or not os.path.exists(target_dir):
        return {"valid": False, "error": "Need input/ and target/ directories"}

    input_files = {Path(f).stem: f for f in os.listdir(input_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    target_files = {Path(f).stem: f for f in os.listdir(target_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

    matched = sorted(set(input_files.keys()) & set(target_files.keys()))
    pairs = []

    for stem in matched:
        inp = os.path.join(input_dir, input_files[stem])
        tgt = os.path.join(target_dir, target_files[stem])
        result = validate_pair(inp, tgt)
        result["name"] = stem
        pairs.append(result)

    # Summary
    scores = [p["score"] for p in pairs]
    grades = [p["grade"] for p in pairs]

    return {
        "valid": len(pairs) > 0,
        "total_pairs": len(pairs),
        "total_input": len(input_files),
        "total_target": len(target_files),
        "unmatched_input": len(input_files) - len(pairs),
        "unmatched_target": len(target_files) - len(pairs),
        "average_score": round(float(np.mean(scores)), 1) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "grade_distribution": {g: grades.count(g) for g in ["A", "B", "C", "D", "F"]},
        "recommended": float(np.mean(scores)) >= 60,
        "pairs": pairs,
    }


def get_recommendation(summary: Dict) -> List[str]:
    """Get recommendations based on validation summary."""
    recs = []

    if summary["unmatched_input"] > 0:
        recs.append(f"{summary['unmatched_input']} input tanpa target — perlu ditambah atau dihapus")

    if summary["unmatched_target"] > 0:
        recs.append(f"{summary['unmatched_target']} target tanpa input — perlu ditambah atau dihapus")

    avg = summary.get("average_score", 0)
    if avg < 50:
        recs.append("Kualitas rendah — perbaiki resolusi, sharpness, dan alignment")
    elif avg < 70:
        recs.append("Kualitas cukup — bisa dipakai tapi ada ruang perbaikan")
    else:
        recs.append("Kualitas bagus — siap untuk training")

    low_pairs = [p for p in summary.get("pairs", []) if p["score"] < 40]
    if low_pairs:
        recs.append(f"{len(low_pairs)} pair score rendah — pertimbangkan hapus atau ambil ulang")

    return recs
