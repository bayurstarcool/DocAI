# AGENTS.md

Instruksi untuk agent yang bekerja di repo `docai`.

## Lingkup
- Berlaku untuk seluruh repo ini kecuali subfolder yang punya `AGENTS.md` sendiri.
- Subproject `shadowremove_repo/` punya instruksi sendiri di `shadowremove_repo/AGENTS.md`; ikuti file itu untuk perubahan di dalam subfolder tersebut.

## Fokus Project
- Project ini Python + Svelte untuk document restoration, terutama shadow removal dan peningkatan kualitas scan/foto dokumen.
- Jalur training utama: `train.py` dengan dataset restoration di `backend/datasets/restoration_dataset.py`.
- UI training utama: `frontend/src/pages/Training.svelte`.
- Backend training API utama: `backend/app.py` endpoint `/api/training/start`.

## Dataset Training
- Tiga tipe dataset yang dipakai UI/training:
  - `paired`: punya `input/` dan `target/` dengan nama file cocok. Ini prioritas utama untuk kualitas model.
  - `clean`: gambar bersih saja. Training membuat degradasi sintetis via `DocumentRestorationDataset`.
  - `identity`: gambar bersih input=target. Dipakai agar model tidak mengubah warna/detail area non-shadow.
- Dataset paired lokal yang saat audit siap dipakai:
  - `datasets/paired/cvpr-2023-rdd`: 4.371 train pairs, 545 test pairs, mask tersedia.
  - `datasets/paired/jungs-dataset`: 67 train pairs, 20 test pairs, mask tersedia.
  - `datasets/paired/kliglers-dataset`: 272 train pairs, 28 test pairs, mask tersedia.
  - `datasets/ShadowDocument7K`: 6.478 train pairs, 760 test pairs, tanpa mask; kode membuat mask dari input-target.
- Total paired siap saat audit: 11.188 train pairs dan 1.353 test/validation pairs.
- Belum ada dataset standar `clean` atau `identity` saat audit.
- Untuk identity, rekomendasi path:
  - `data/identity/` untuk gambar bersih langsung.
  - `datasets/identity/<nama_dataset>/` untuk beberapa kumpulan identity.
- Tambahkan 20–100 gambar dokumen bersih ke identity jika output model mulai mengubah warna area tanpa shadow.

## Rekomendasi Training
- Lewat UI, pilih semua dataset `paired` yang ready untuk baseline kuat.
- Jangan skip validation; UI otomatis memakai split `test` dari dataset paired jika tersedia.
- Default yang disarankan untuk NVIDIA L4:
  - `size=768`
  - `batch_size=4`
  - `lr=1e-4`
  - `early_stop_patience=7`
  - `grad_clip_norm=1.0`
- Untuk fine-tune kualitas, gunakan `resume_weights_only` dari `checkpoints/document_restorer/best.pth`, LR kecil, dan validation nyata bila ada.
- Jangan menjalankan training berat lokal tanpa konfirmasi; prefer GPU/Camber/L4.

## UI/API Sinkronisasi
- Jika menambah argumen baru ke `train.py`, sinkronkan juga:
  - form parameter di `backend/app.py` endpoint `/api/training/start`.
  - payload di `frontend/src/pages/Training.svelte`.
  - kontrol UI jika user perlu memilih nilainya.
- Dataset selection UI harus mengirim:
  - `paired` ke `paired_data`.
  - `clean` ke `clean_data`.
  - `identity` ke `identity_data`.
- Setelah mengubah frontend, jalankan `cd frontend && npm run build` bila memungkinkan.
- Setelah mengubah Python, minimal jalankan `python3 -m py_compile <file>`.
