from __future__ import annotations
import argparse
from pathlib import Path
import random
import sys
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synth_shadow import apply_document_shadow, apply_object_shadow

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def list_images(root: str | Path):
    root = Path(root)
    return sorted([path for path in root.rglob('*') if path.suffix.lower() in IMG_EXTS])


def imread_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description='Generate aligned static shadow/clean/mask pairs from clean documents.')
    parser.add_argument('--clean', required=True, help='Directory containing clean document images.')
    parser.add_argument('--out', default='data/static_shadow_pairs', help='Output paired dataset directory.')
    parser.add_argument('--copies', type=int, default=8, help='Static shadow variants per clean image.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible static pairs.')
    parser.add_argument('--style', choices=['document', 'object', 'mixed'], default='object', help='Shadow synthesis style.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    clean_paths = list_images(args.clean)
    if not clean_paths:
        raise RuntimeError(f'No clean images found: {args.clean}')

    out = Path(args.out)
    shadow_dir = out / 'shadow'
    clean_dir = out / 'clean'
    mask_dir = out / 'mask'
    shadow_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for clean_path in tqdm(clean_paths, desc='static shadow pairs'):
        clean = imread_rgb(clean_path)
        for variant in range(args.copies):
            if args.style == 'object':
                shadow, mask = apply_object_shadow(clean, True)
            elif args.style == 'mixed' and variant % 3 == 0:
                shadow, mask = apply_document_shadow(clean, True)
            else:
                shadow, mask = apply_object_shadow(clean, True)
            name = f'{clean_path.stem}_{variant:03d}.png'
            imwrite_rgb(shadow_dir / name, shadow)
            imwrite_rgb(clean_dir / name, clean)
            cv2.imwrite(str(mask_dir / name), mask)
            count += 1

    print(f'saved {count} aligned pairs to {out}')
    print(f'shadow: {shadow_dir}')
    print(f'clean:  {clean_dir}')
    print(f'mask:   {mask_dir}')


if __name__ == '__main__':
    main()
