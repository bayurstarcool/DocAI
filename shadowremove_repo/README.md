# Doc Shadow AI v2

Pipeline AI untuk document enhancement: deteksi shadow, remove shadow, output shadow mask, dan filter OCR-ready.

## Install tanpa Docker

```bash
conda create -n docai python=3.11 -y
conda activate docai
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Inference tanpa model AI

```bash
python infer.py input.jpg output.jpg --no-ai --mode magic
```

## Training synthetic dari clean documents

Masukkan gambar dokumen bersih ke `data/clean_docs`, lalu:

```bash
python train.py --data data/clean_docs --epochs 80 --batch 16 --size 1024 --amp
```

Dengan GPU VRAM besar:

```bash
python train.py --data data/clean_docs --epochs 120 --batch 32 --size 1536 --workers 16 --amp
```

## Training di CamberCloud

CamberCloud public docs saat ini menyediakan on-demand GPU NVIDIA L4. Gunakan wheel PyTorch CUDA, bukan ROCm.

Setup environment:

```bash
bash scripts/camber/setup_camber.sh
```

Training preset L4:

```bash
bash scripts/camber/train_camber_l4.sh
```

Override ringan lewat env:

```bash
DATA_DIR=data/clean_docs OUT_DIR=runs/camber_l4 EPOCHS=80 BATCH=4 SIZE=1024 WORKERS=4 bash scripts/camber/train_camber_l4.sh
```

Dataset tidak ikut Git. Upload/sync dataset ke `data/clean_docs` atau struktur paired dataset sebelum training.

Dataset training final yang direkomendasikan untuk CamberCloud adalah paired static-object-shadow:

```text
stash://bayurstarcool/projects/shadowremove/data/static_pairs/shadow
stash://bayurstarcool/projects/shadowremove/data/static_pairs/clean
stash://bayurstarcool/projects/shadowremove/data/static_pairs/mask
```

Dataset ini dibuat dari `datasetku/cleans_rgb`, sehingga clean/shadow/mask selalu aligned.

### CamberCloud CLI

Setelah login `camber login`, upload project kecil ke Stash tanpa dataset besar:

```bash
source ~/.zshrc
bash scripts/camber/preflight.sh
STASH_PATH=stash://bayurstarcool/projects/shadowremove/ bash scripts/camber/upload_to_stash.sh
```

Upload dataset clean kecil/pribadi ke Stash untuk synthetic shadow training:

```bash
DATA_SRC=datasetku/cleans_rgb DEST=stash://bayurstarcool/projects/shadowremove/data/clean_docs/ bash scripts/camber/upload_clean_dataset.sh
```

Atau generate lalu upload paired dataset clean + shadow statis yang aligned. Ini wajib kalau clean/shadow manual beda ukuran/pixel:

```bash
python scripts/make_static_shadow_pairs.py --clean datasetku/cleans_rgb --out data/static_shadow_pairs --copies 8 --seed 42 --style object
python scripts/validate_pairs.py --shadow data/static_shadow_pairs/shadow --clean data/static_shadow_pairs/clean --mask data/static_shadow_pairs/mask
SHADOW_SRC=data/static_shadow_pairs/shadow CLEAN_SRC=data/static_shadow_pairs/clean MASK_SRC=data/static_shadow_pairs/mask BASE_DEST=stash://bayurstarcool/projects/shadowremove/data/static_pairs/ bash scripts/camber/upload_paired_dataset.sh
```

Buat GPU job synthetic clean-data dari Stash:

```bash
STASH_PATH=stash://bayurstarcool/projects/shadowremove/ MODE=clean SIZE=small DATA_DIR=data/clean_docs OUT_DIR=runs/camber_l4 EPOCHS=80 BATCH=4 IMG_SIZE=1024 WORKERS=4 bash scripts/camber/create_train_job.sh
```

Buat GPU job paired-data:

```bash
STASH_PATH=stash://bayurstarcool/projects/shadowremove/ MODE=paired SIZE=xsmall PAIRED_SHADOW=data/static_pairs/shadow PAIRED_CLEAN=data/static_pairs/clean PAIRED_MASK=data/static_pairs/mask OUT_DIR=runs/camber_l4_paired_xsmall_768 EPOCHS=80 BATCH=1 IMG_SIZE=768 WORKERS=0 BASE=32 bash scripts/camber/create_train_job.sh
```

Cek job dan log:

```bash
camber job list
camber job logs <job-id>
camber job get <job-id>
```

Setelah selesai, output checkpoint ada di Stash project mount:

```text
stash://bayurstarcool/projects/shadowremove/runs/camber_l4_paired_xsmall_768/
```

Tarik output ke lokal:

```bash
RUN_NAME=camber_l4_paired_xsmall_768 SRC=stash://bayurstarcool/projects/shadowremove/runs/camber_l4_paired_xsmall_768/ DEST=runs/camber_l4_paired_xsmall_768 bash scripts/camber/pull_run_output.sh
```

## Training paired dataset

Struktur:

```text
data/sd7k/shadow/*.jpg
data/sd7k/clean/*.jpg
data/sd7k/mask/*.png optional
```

Run:

```bash
python train.py --paired-shadow data/sd7k/shadow --paired-clean data/sd7k/clean --paired-mask data/sd7k/mask --epochs 120 --batch 16 --size 1024
```

## SD7K OpenXLab subset di CamberCloud

Untuk dataset besar `lkljty/ShadowDocument7K`, jangan download lokal dan jangan download full dulu. Gunakan job Camber untuk selective subset:

```bash
STASH_PATH=stash://bayurstarcool/projects/shadowremove/ PAIR_LIMIT=200 OUT_DIR=data/sd7k_subset bash scripts/camber/create_sd7k_prepare_job.sh
```

Credential OpenXLab harus disimpan sebagai file privat di Stash project mount:

```text
stash://bayurstarcool/projects/shadowremove/secrets/openxlab_config.json
```

Format file:

```json
{"ak":"<OPENXLAB_ACCESS_KEY_ID>","sk":"<OPENXLAB_SECRET_ACCESS_KEY>"}
```

File credential jangan commit dan jangan taruh di command job karena command muncul di `camber job list`.

Output subset:

```text
stash://bayurstarcool/projects/shadowremove/data/sd7k_subset/shadow
stash://bayurstarcool/projects/shadowremove/data/sd7k_subset/clean
stash://bayurstarcool/projects/shadowremove/data/sd7k_subset/mask
```

## Generate dataset sintetis

```bash
python scripts/make_synthetic_pairs.py --clean data/clean_docs --out data/synth_pairs --copies 10
```

Untuk paired dataset statis yang clean/shadow/mask selalu sama pixel size dan nama file:

```bash
python scripts/make_static_shadow_pairs.py --clean datasetku/cleans_rgb --out data/static_shadow_pairs --copies 8 --seed 42 --style object
```

## Inference AI

```bash
python infer.py input.jpg enhanced.jpg --weights runs/doc_enhancer/best.pth --mode ai_magic --save-mask shadow_mask.png --save-json meta.json
```

## API

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
```

Upload:

```bash
curl -F "file=@input.jpg" -F "mode=ai_magic" http://localhost:8000/enhance --output enhanced.jpg
```

## Export ONNX

```bash
mkdir -p exports
python export_onnx.py --weights runs/doc_enhancer/best.pth --out exports/doc_enhancer.onnx --size 1024
```
