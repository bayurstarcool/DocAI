#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO="${DATASET_REPO:-lkljty/ShadowDocument7K}"
PAIR_LIMIT="${PAIR_LIMIT:-200}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-sd7k_subset}"
WORK_DIR="${WORK_DIR:-sd7k_work}"
SECRET_CONFIG="${SECRET_CONFIG:-secrets/openxlab_config.json}"
MANIFEST_DIR="${MANIFEST_DIR:-sd7k_manifest}"

mkdir -p "$OUT_DIR/shadow" "$OUT_DIR/clean" "$OUT_DIR/mask" "$WORK_DIR" "$MANIFEST_DIR" ~/.openxlab

python -m pip install -q -U openxlab rich

python - <<'PY'
from pathlib import Path
import openxlab

root = Path(openxlab.__file__).resolve().parent
target = root / 'dataset' / 'handler' / 'list_dataset_repository.py'
text = target.read_text()
if 'rprint(' in text and 'from rich import print as rprint' not in text:
    target.write_text('from rich import print as rprint\n' + text)
PY

if [ -n "${OPENXLAB_AK:-}" ] && [ -n "${OPENXLAB_SK:-}" ]; then
  python - <<'PY'
import json, os
from pathlib import Path
Path.home().joinpath('.openxlab').mkdir(exist_ok=True)
Path.home().joinpath('.openxlab/config.json').write_text(json.dumps({'ak': os.environ['OPENXLAB_AK'], 'sk': os.environ['OPENXLAB_SK']}))
PY
elif [ -f "$SECRET_CONFIG" ]; then
  cp "$SECRET_CONFIG" ~/.openxlab/config.json
else
  echo "Missing OpenXLab credentials. Set OPENXLAB_AK/OPENXLAB_SK or provide $SECRET_CONFIG" >&2
  exit 2
fi
chmod 600 ~/.openxlab/config.json

echo "Listing OpenXLab dataset: $DATASET_REPO"
openxlab dataset ls --dataset-repo "$DATASET_REPO" > "$MANIFEST_DIR/openxlab_ls.txt"

python - <<'PY'
from __future__ import annotations
import os, random, re, subprocess, shutil
from pathlib import Path

repo=os.environ.get('DATASET_REPO','lkljty/ShadowDocument7K')
limit=int(os.environ.get('PAIR_LIMIT','200'))
seed=int(os.environ.get('SEED','42'))
out=Path(os.environ.get('OUT_DIR','sd7k_subset'))
work=Path(os.environ.get('WORK_DIR','sd7k_work'))
manifest=Path(os.environ.get('MANIFEST_DIR','sd7k_manifest'))
text=(manifest/'openxlab_ls.txt').read_text(errors='ignore')

exts=r'\.(?:jpg|jpeg|png|webp|bmp|tif|tiff)'
paths=[]
for token in re.split(r'\s+', text):
    token=token.strip().strip('|').strip(',')
    if re.search(exts+r'$', token, re.I):
        paths.append(token)
paths=sorted(set(paths))
if not paths:
    print(text[:4000])
    raise SystemExit('No image paths parsed from openxlab dataset ls output. Inspect data/sd7k_manifest/openxlab_ls.txt')

shadow_keys=('shadow','input','shad')
mask_keys=('mask','matte')
clean_keys=('clean','shadow_free','shadow-free','shadowfree','gt','target','free')

def kind(path:str):
    low=path.lower().replace(' ', '_')
    if any(k in low for k in mask_keys): return 'mask'
    if any(k in low for k in clean_keys) and 'shadow' not in Path(low).name: return 'clean'
    if any(k in low for k in shadow_keys): return 'shadow'
    return 'unknown'

def pair_key(path:str):
    stem=Path(path).stem.lower()
    stem=re.sub(r'(?:_?shadow|_?clean|_?mask|_?gt|_?shadowfree|_?shadow_free|_?shadow-free)$','',stem)
    stem=re.sub(r'[^a-z0-9]+','_',stem).strip('_')
    return stem

buckets={}
for path in paths:
    k=kind(path)
    if k=='unknown':
        continue
    key=pair_key(path)
    buckets.setdefault(key,{})[k]=path
pairs=[]
for key,item in buckets.items():
    if 'shadow' in item and 'clean' in item:
        pairs.append((key,item.get('shadow'),item.get('clean'),item.get('mask')))

(manifest/'parsed_paths.txt').write_text('\n'.join(paths))
(manifest/'parsed_pairs.tsv').write_text('\n'.join('\t'.join(str(x or '') for x in row) for row in pairs))

if not pairs:
    raise SystemExit('No shadow/clean pairs detected. Inspect data/sd7k_manifest/parsed_paths.txt and parsed_pairs.tsv')

random.Random(seed).shuffle(pairs)
pairs=pairs[:limit]
print(f'Detected {len(buckets)} keys, selected {len(pairs)} pairs of requested {limit}')

for folder in ['shadow','clean','mask']:
    (out/folder).mkdir(parents=True, exist_ok=True)

def download_one(src:str, dst_dir:Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    before=set(p for p in dst_dir.rglob('*') if p.is_file())
    cmd=['openxlab','dataset','download','--dataset-repo',repo,'--source-path',src,'--target-path',str(dst_dir)]
    subprocess.run(cmd, check=True)
    after=set(p for p in dst_dir.rglob('*') if p.is_file())
    new=sorted(after-before, key=lambda p:p.stat().st_mtime, reverse=True)
    if new:
        return new[0]
    candidates=sorted(dst_dir.rglob(Path(src).name))
    if candidates:
        return candidates[-1]
    raise RuntimeError(f'Downloaded file not found for {src}')

for index,(key,shadow_src,clean_src,mask_src) in enumerate(pairs):
    name=f'{index:05d}_{key}.png'
    shadow_dst=out/'shadow'/name
    clean_dst=out/'clean'/name
    mask_dst=out/'mask'/name
    if shadow_dst.exists() and clean_dst.exists() and (not mask_src or mask_dst.exists()):
        print(f'Skip existing pair {index+1}/{len(pairs)}: {name}', flush=True)
        continue
    print(f'Downloading pair {index+1}/{len(pairs)}: {name}', flush=True)
    tmp=work/'downloads'/f'{index:05d}'
    shadow_file=download_one(shadow_src, tmp/'shadow')
    clean_file=download_one(clean_src, tmp/'clean')
    shutil.copy2(shadow_file, shadow_dst)
    shutil.copy2(clean_file, clean_dst)
    if mask_src:
        mask_file=download_one(mask_src, tmp/'mask')
        shutil.copy2(mask_file, mask_dst)

print(f'Saved subset to {out}')
print(f'Shadow files: {len(list((out/"shadow").glob("*")))}')
print(f'Clean files: {len(list((out/"clean").glob("*")))}')
print(f'Mask files: {len(list((out/"mask").glob("*")))}')
PY
