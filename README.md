# 📄 DocAI

**DocAI** adalah proyek yang berfokus pada pengembangan model AI canggih untuk membersihkan dan meningkatkan kualitas gambar dokumen yang dipindai. Tujuannya adalah untuk menghasilkan gambar yang jernih, bebas bayangan, dan mudah dibaca dari input yang mungkin memiliki kualitas rendah.

Proyek ini terdiri dari dua komponen utama:
1.  **Aplikasi Web Sederhana:** Antarmuka yang mudah digunakan untuk menguji model pada gambar Anda sendiri.
2.  **Pipeline Training:** Skrip dan utilitas untuk melatih model AI dari awal menggunakan dataset Anda.

---

## 🚀 Memulai

### 1. Instalasi Dependensi

Pastikan Anda memiliki Python 3.8+ dan pip. Disarankan untuk menggunakan virtual environment.

```bash
# Buat virtual environment (opsional tapi disarankan)
python3 -m venv venv
source venv/bin/activate

# Install semua library yang dibutuhkan
pip install -r requirements.txt
```

### 2. Menjalankan Aplikasi Web

Setelah instalasi selesai, Anda dapat menjalankan server web.

```bash
# Jalankan server menggunakan skrip start.sh
./start.sh
```

Aplikasi akan tersedia di **http://0.0.0.0:8000**.

**Catatan Penting:** Secara default, aplikasi web mungkin tidak memiliki model untuk dijalankan. Jika file `checkpoints/document_restorer/best.pth` tidak ada, server akan berjalan tetapi tidak akan bisa memproses gambar. Anda harus melatih model terlebih dahulu.

---

## 🎓 Melatih Model AI Anda

Proses training dipisahkan dari aplikasi web dan dijalankan melalui command line.

### 1. Siapkan Dataset

Proyek ini dirancang untuk bekerja dengan dataset berpasangan, di mana Anda memiliki gambar dokumen yang "buruk" (input) dan versi "bersih" (target). Format yang direkomendasikan adalah seperti dataset **ShadowDocument7K (SD7K)**.

*   Letakkan dataset Anda di dalam direktori `data/`.
*   Struktur yang diharapkan:
    ```
    data/
    └── paired_dataset/
        ├── train/
        │   ├── input/
        │   │   ├── 0001.jpg
        │   │   └── ...
        │   └── target/
        │       ├── 0001.png
        │       └── ...
        └── test/
            ├── input/
            │   └── ...
            └── target/
                └── ...
    ```

### 2. Jalankan Skrip Training

Gunakan skrip `train.py` untuk memulai proses training.

```bash
python train.py \
  --paired-data data/paired_dataset \
  --output checkpoints/docai_v1 \
  --epochs 100 \
  --batch-size 8 \
  --size 512 \
  --lr 2e-4
```

- `--paired-data`: Path ke direktori dataset berpasangan Anda.
- `--output`: Direktori tempat menyimpan checkpoint model.
- `--epochs`: Jumlah epoch training.
- `--batch-size`: Ukuran batch (sesuaikan dengan VRAM GPU Anda).
- `--size`: Ukuran gambar yang akan digunakan saat training (misal: 512x512).
- `--lr`: Learning rate.

Checkpoint model terbaik akan disimpan sebagai `best.pth` di dalam direktori output yang Anda tentukan.

### 3. Validasi Cepat Test dan Training

Jalankan unit test tanpa dependency tambahan seperti `pytest`:

```bash
PYTHONPATH=$(pwd) python3 -m unittest discover -s tests -v
```

Untuk smoke test training kecil di CPU, gunakan subset kecil dari dataset berpasangan:

```bash
PYTHONPATH=$(pwd) python3 train.py \
  --paired-data data/paired_dataset \
  --output /tmp/docai_train_smoke \
  --epochs 1 \
  --batch-size 1 \
  --size 128 \
  --workers 0 \
  --base-channels 8 \
  --max-train-samples 2 \
  --max-validation-samples 1 \
  --device cpu
```

### 4. Gunakan Model Baru Anda

Setelah training selesai, salin model terbaik ke lokasi yang digunakan oleh aplikasi web:

```bash
# Buat direktori jika belum ada
mkdir -p checkpoints/document_restorer

# Salin model baru Anda
cp checkpoints/docai_v1/best.pth checkpoints/document_restorer/best.pth
```

Setelah itu, **restart server web** (`./start.sh`), dan aplikasi akan secara otomatis menggunakan model baru yang telah Anda latih.

---

## 🏛️ Struktur Proyek yang Disederhanakan

```
DocAI/
├── backend/
│   └── app.py              # Aplikasi web FastAPI yang sederhana
├── frontend/
│   └── templates/
│       └── index.html      # Halaman antarmuka web
├── data/                   # Direktori untuk menyimpan dataset
├── checkpoints/            # Direktori untuk menyimpan model hasil training
├── src/
│   ├── dataset.py          # Logika pemuatan data
│   └── model.py            # Arsitektur model DocumentRestorerNet
├── train.py                # Skrip utama untuk training model
├── requirements.txt        # Daftar dependensi Python
├── start.sh                # Skrip untuk menjalankan server web
└── README.md               # File ini
```
