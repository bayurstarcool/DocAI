from __future__ import annotations
import argparse
from pathlib import Path
import cv2

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def list_images(root: str | Path):
    root = Path(root)
    return sorted([path for path in root.rglob('*') if path.suffix.lower() in IMG_EXTS])


def image_size(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    return width, height


def main():
    parser = argparse.ArgumentParser(description='Validate paired shadow/clean/mask dataset alignment.')
    parser.add_argument('--shadow', required=True)
    parser.add_argument('--clean', required=True)
    parser.add_argument('--mask')
    args = parser.parse_args()

    shadow_paths = list_images(args.shadow)
    clean_root = Path(args.clean)
    mask_root = Path(args.mask) if args.mask else None
    errors = []

    if not shadow_paths:
        raise RuntimeError(f'No shadow images found: {args.shadow}')

    for shadow_path in shadow_paths:
        clean_path = clean_root / shadow_path.name
        mask_path = mask_root / shadow_path.name if mask_root else None
        if not clean_path.exists():
            errors.append(f'missing clean: {clean_path}')
            continue
        shadow_size = image_size(shadow_path)
        clean_size = image_size(clean_path)
        if shadow_size != clean_size:
            errors.append(f'size mismatch: {shadow_path.name} shadow={shadow_size} clean={clean_size}')
        if mask_path:
            if not mask_path.exists():
                errors.append(f'missing mask: {mask_path}')
            else:
                mask_size = image_size(mask_path)
                if shadow_size != mask_size:
                    errors.append(f'mask size mismatch: {shadow_path.name} shadow={shadow_size} mask={mask_size}')

    clean_names = {path.name for path in list_images(clean_root)}
    shadow_names = {path.name for path in shadow_paths}
    extra_clean = sorted(clean_names - shadow_names)
    if extra_clean:
        errors.append(f'extra clean files: {len(extra_clean)}')

    if errors:
        for error in errors[:50]:
            print(error)
        raise SystemExit(f'pair validation failed: {len(errors)} error(s)')

    print(f'pair validation ok: {len(shadow_paths)} items')


if __name__ == '__main__':
    main()
