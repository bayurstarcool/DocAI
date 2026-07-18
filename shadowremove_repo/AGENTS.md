# AGENTS.md

Instruksi untuk agent yang bekerja di repo ini.

## Lingkup
- Berlaku untuk seluruh repo `doc_shadow_ai`.
- Project ini Python/AI untuk deteksi atau perbaikan shadow dokumen, dengan entrypoint utama `train.py`, `infer.py`, `export_onnx.py`, dan API di `api/`.

## Environment Utama
- Jalur training utama saat ini: CamberCloud GPU NVIDIA L4/CUDA.
- Lokal dipakai untuk edit, audit ringan, Web UI CPU, dan pull checkpoint.
- Jangan asumsikan laptop lokal cukup untuk training; inference lokal CPU bisa lambat sekitar 9–30 detik per gambar besar.
- VPS lama `root@134.199.195.79` pernah dipakai, tetapi server pernah destroy/hilang; jangan jadikan source of truth tanpa audit ulang.
- Jika perlu cek VPS lagi, akses SSH memakai key tanpa password/passphrase interaktif.
- Source of truth repo: GitHub remote `git@github.com:bayurstarcool/shadowremove.git`, branch `main`.
- Source of truth artefak training: Camber Stash `stash://bayurstarcool/projects/shadowremove/`.

## Domain, Nginx, dan Web Dev
- Status domain/VPS bisa berubah karena server pernah destroy; audit ulang sebelum mengandalkan bagian ini.
- Domain public untuk training/dev UI: `https://train.sigarda.com`.
- Nginx aktif di VPS dan config domain ada di `/etc/nginx/conf.d/train.sigarda.com.conf`.
- Config saat audit: `listen 443 ssl`, `server_name train.sigarda.com`, proxy ke `http://127.0.0.1:80`.
- Aplikasi FastAPI/Uvicorn remote berjalan dari `/root/doc_shadow_ai` memakai `.venv-rocm`, binding `0.0.0.0:80`.
- Endpoint root `/` mengembalikan `404` karena API tidak punya route root; ini bukan indikator service mati.
- UI utama ada di `/train-ui/`, `/dataset-ui/`, dan `/studio/`.
- Semua UI/API admin memakai HTTP Basic Auth dari env remote `.admin_env`; jangan tampilkan atau commit nilai `ADMIN_USER` / `ADMIN_PASSWORD`.
- Health publik tanpa auth: `https://train.sigarda.com/health`.
- Sebelum ubah Nginx, jalankan `nginx -t`; setelah valid baru `systemctl reload nginx`.

## SSH dan Remote
- Untuk akses VPS, gunakan SSH outbound:
  - `ssh root@134.199.195.79`
- Jika approval/sandbox mode mengizinkan escalation, gunakan prefix rule SSH yang aman, misalnya `prefix_rule: ["ssh"]`.
- Jika approval policy `never`, jangan set `sandbox_permissions`; command akan ditolak oleh harness.
- Jangan hardcode private key, token, credential, atau secret ke repo.
- Sebelum operasi berat remote, cek kondisi server dulu:
  - `hostname`
  - `pwd`
  - `df -h`
  - `free -h`
  - `rocm-smi` atau tool GPU AMD lain jika tersedia
  - `python3 --version`
  - `docker --version` jika workflow Docker dipakai
  - `systemctl is-active nginx`
  - `ss -ltnp` untuk cek port `80` / `443`

## Sinkronisasi Lokal ke VPS
- Sebelum resync, audit dulu file besar dan file sensitif lokal.
- Target sync default: `root@134.199.195.79:/root/doc_shadow_ai/`.
- Jangan sync artefak cache/build:
  - `.git/`
  - `__pycache__/`
  - `*.pyc`
  - `.DS_Store`
  - virtualenv seperti `.venv/`, `venv/`
  - output model/checkpoint besar kecuali memang diminta
- Dataset besar di `datasetku/` jangan otomatis dikirim tanpa konfirmasi eksplisit.
- Artefak remote besar seperti `data/`, `models/`, `runs/`, `exports/`, `.venv/`, `.venv-rocm/`, `.miopen/`, dan log training jangan ditimpa/hapus tanpa konfirmasi eksplisit.
- Prefer `rsync` dengan exclude jelas untuk sinkronisasi ke server.
- Lakukan dry-run dulu sebelum sync destruktif atau besar:
  - `rsync -avzn --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' --exclude '.venv/' --exclude 'venv/' --exclude 'datasetku/' --exclude 'data/' --exclude 'models/' --exclude 'runs/' --exclude 'exports/' ./ root@134.199.195.79:/root/doc_shadow_ai/`
- Setelah dry-run aman, baru jalankan sync nyata jika user setuju.
- Hindari `--delete` untuk sync awal kecuali user eksplisit minta dan dry-run sudah ditinjau.

## Python / AI Workflow
- Prefer validasi ringan lokal dulu:
  - import check
  - syntax check
  - unit/inference kecil jika data tersedia
- Untuk training, benchmark, atau proses GPU, prefer CamberCloud L4. VPS AMD hanya dipakai jika sudah diaudit ulang dan ROCm valid.
- Untuk cek ROCm remote:
  - `ssh root@134.199.195.79 'cd /root/doc_shadow_ai && .venv-rocm/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"'`
- Jangan menjalankan training berat tanpa konfirmasi user.
- Jangan menambahkan dependency baru tanpa alasan kuat dan update `requirements.txt`.

## Testing dan Build
- Project ini dominan Python; validasi cepat minimal: `python3 -m py_compile <file>` untuk file Python yang diubah.
- Untuk Web UI lokal, cek `curl -s http://127.0.0.1:8000/health` setelah restart Uvicorn.
- Untuk endpoint enhance, pakai request kecil dulu agar CPU lokal tidak berat.
- Jika ada subproject React/Node, jangan pakai `npm run build` atau command build berat; prefer `npx tsc --noEmit --incremental false`.
- Jika ada backend Go, testing ringan prefer via `curl` ke service berjalan dan hindari proses berat tanpa perlu.

## CamberCloud
- CamberCloud adalah jalur training utama saat ini jika instance menyediakan GPU NVIDIA L4/CUDA.
- Preset CamberCloud ada di `configs/camber/train_l4.yaml`.
- Setup environment CamberCloud: `bash scripts/camber/setup_camber.sh`.
- Run training CamberCloud L4: `bash scripts/camber/train_camber_l4.sh`.
- Upload project ke Stash: `STASH_PATH=stash://bayurstarcool/projects/shadowremove/ bash scripts/camber/upload_to_stash.sh`.
- Preflight Camber: `bash scripts/camber/preflight.sh`.
- Upload clean dataset: `DATA_SRC=datasetku/cleans_rgb DEST=stash://bayurstarcool/projects/shadowremove/data/clean_docs/ bash scripts/camber/upload_clean_dataset.sh`.
- Upload paired dataset: `SHADOW_SRC=datasetku/shadow CLEAN_SRC=datasetku/cleans_rgb BASE_DEST=stash://bayurstarcool/projects/shadowremove/data/user_pairs/ bash scripts/camber/upload_paired_dataset.sh`.
- Untuk paired training, jangan pakai clean/shadow manual yang beda ukuran/alignment. Generate dulu shadow object/tangan statis aligned: `python scripts/make_static_shadow_pairs.py --clean datasetku/cleans_rgb --out data/static_shadow_pairs --copies 8 --seed 42 --style object`.
- Upload paired aligned ke path bersih: `SHADOW_SRC=data/static_shadow_pairs/shadow CLEAN_SRC=data/static_shadow_pairs/clean MASK_SRC=data/static_shadow_pairs/mask BASE_DEST=stash://bayurstarcool/projects/shadowremove/data/static_pairs/ bash scripts/camber/upload_paired_dataset.sh`.
- Validasi paired dataset sebelum upload/train: `python scripts/validate_pairs.py --shadow data/static_shadow_pairs/shadow --clean data/static_shadow_pairs/clean --mask data/static_shadow_pairs/mask`.
- Dataset synthetic aligned final: `stash://bayurstarcool/projects/shadowremove/data/static_pairs/`.
- Dataset SD7K test subset final: `stash://bayurstarcool/projects/shadowremove/sd7k_subset_20/` berisi `20` shadow/clean/mask pair.
- Dataset SD7K partial 200 pernah dicoba: `stash://bayurstarcool/projects/shadowremove/sd7k_subset_200/`, saat cancel hanya sekitar `13` pair; jangan anggap lengkap.
- Create clean GPU job: `STASH_PATH=stash://bayurstarcool/projects/shadowremove/ MODE=clean SIZE=small bash scripts/camber/create_train_job.sh`.
- Create paired GPU job xsmall: `STASH_PATH=stash://bayurstarcool/projects/shadowremove/ MODE=paired SIZE=xsmall PAIRED_SHADOW=data/static_pairs/shadow PAIRED_CLEAN=data/static_pairs/clean PAIRED_MASK=data/static_pairs/mask BATCH=1 IMG_SIZE=768 BASE=32 WORKERS=0 bash scripts/camber/create_train_job.sh`.
- Camber GPU xsmall punya `/dev/shm` terbatas; gunakan `WORKERS=0` untuk menghindari DataLoader bus error.
- Camber L4 xsmall OOM di `IMG_SIZE=1024`, `BATCH=2`, `BASE=48`; default retry aman: `IMG_SIZE=768`, `BATCH=1`, `BASE=32`, `WORKERS=0`.
- Training output otomatis tersimpan di Stash project mount jika `OUT_DIR` berada di `runs/...`; jangan panggil `camber` di dalam worker job karena CLI tidak tersedia.
- Pull output lokal via `camber stash cp` bisa membuat file placeholder `1B`; untuk checkpoint besar lebih aman pakai `camber stash presign-url --file ...` lalu `curl -L`, atau range download paralel jika koneksi lambat.
- Pull output lokal contoh lama: `RUN_NAME=camber_l4_paired_xsmall_768 SRC=stash://bayurstarcool/projects/shadowremove/runs/camber_l4_paired_xsmall_768/ DEST=runs/camber_l4_paired_xsmall_768 bash scripts/camber/pull_run_output.sh`.
- CamberCloud L4 memakai CUDA PyTorch wheel (`cu124`), bukan ROCm.
- Jangan sync atau commit dataset besar; upload dataset ke storage/runtime CamberCloud lalu arahkan `DATA_DIR`.
- Default aman Camber L4 xsmall: `BATCH=1`, `IMG_SIZE=768`, `BASE=32`, `WORKERS=0`.
- Jangan create job GPU tanpa konfirmasi user karena berpotensi biaya.
- Untuk SD7K OpenXLab besar, gunakan selective subset job dan path writable root Stash, bukan `data/...`: `PAIR_LIMIT=200 OUT_DIR=sd7k_subset_200 WORK_DIR=sd7k_work_200 MANIFEST_DIR=sd7k_manifest_200 bash scripts/camber/create_sd7k_prepare_job.sh`.
- Credential OpenXLab harus disimpan privat di Stash `secrets/openxlab_config.json`, tidak boleh masuk Git dan tidak boleh muncul di command job.
- Jangan download full `lkljty/ShadowDocument7K` ke lokal.

## Model dan Hasil Terbaru
- Model aktif lokal terbaik saat ini: `runs/combined_static_sd7k_l4_xsmall_768/best.pth`.
- Model ini berasal dari Camber job `22866`, output Stash: `stash://bayurstarcool/projects/shadowremove/runs/combined_static_sd7k_l4_xsmall_768/`.
- Hasil training gabungan job `22866`: `epoch=80 train_loss=0.0713 val_psnr=31.47 best=31.61`.
- Pembanding static-only job `22865`: `epoch=80 train_loss=0.0553 val_psnr=30.04 best=30.04`.
- Model gabungan lebih baik secara PSNR, tetapi hasil visual masih belum produksi: shadow besar masih tersisa, teks bisa soft, warna bisa berubah jika postprocess terlalu agresif.
- `.admin_env` lokal diarahkan ke model gabungan: `MODEL_PATH=runs/combined_static_sd7k_l4_xsmall_768/best.pth`.
- Web UI lokal: `http://127.0.0.1:8000/studio/` dengan Basic Auth dari `.admin_env`.
- Start lokal: `set -a; . ./.admin_env; set +a; .venv-local/bin/uvicorn api.server:app --host 127.0.0.1 --port 8000 --workers 1`.

## Inference dan Postprocess
- `api.server.ai_enhance()` memakai model AI lalu `src.enhance.ai_shadow_postprocess()`.
- Postprocess harus menjaga warna dokumen: AI dominan hanya memengaruhi luminance/lightness, sedangkan chroma/color original dipertahankan.
- Jangan menambah CLAHE global/RGB blend agresif pada output AI karena membuat warna berubah kuning/kontras dan tekstur kertas berlebihan.
- Detail teks dijaga dengan edge/detail restore ringan; jangan over-sharpen sampai noise kertas naik.
- Mode evaluasi visual utama: `AI only`. Mode `AI Magic` bisa lebih putih, tapi risiko warna/teks berubah lebih besar.
- Jika user melaporkan warna berubah, cek `src/enhance.py` bagian `ai_shadow_postprocess()` sebelum menyalahkan model.

## Docker
- Ada `Dockerfile` dan `docker-compose.yml`; cek isinya sebelum memakai Docker.
- Jangan build image besar tanpa konfirmasi user.
- Jika memakai Docker di VPS GPU AMD, pastikan dukungan ROCm/container runtime tersedia dulu.

## Git dan File Safety
- Jangan commit, branch, reset, atau push kecuali user minta eksplisit.
- Jangan hapus dataset/model/checkpoint tanpa konfirmasi eksplisit.
- Jaga perubahan minimal dan fokus pada request.
- Pakai `rg` / `rg --files` untuk pencarian cepat.

## Bahasa Respons
- User memakai Bahasa Indonesia; jawab Bahasa Indonesia kecuali diminta lain.
- Ringkas, langsung, dan sebut path/command persis saat relevan.
