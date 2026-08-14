# Open-source Dataset Inventory

Server: `38.47.85.224`
Audit date: `2026-08-14`

Open-source datasets are intentionally **not copied** to emergency backup. Re-download using URLs below. Keep project `README.md`, `AGENTS.md`, `DATASETS.md`, and `DATASET_DOWNLOAD.md` with source code.

## ShadowDocument7K / SD7K

- Local path: `datasets/ShadowDocument7K`
- Expected layout: `train/input`, `train/target`, `test/input`, `test/target`
- Source page: <https://openxlab.org.cn/datasets/lkljty/ShadowDocument7K>
- GitHub: <https://github.com/CXH-Research/DocShadow-SD7K>
- Paper: <https://arxiv.org/abs/2308.14221>
- Download instructions: `DATASET_DOWNLOAD.md`, `DOWNLOAD_INSTRUCTIONS.md`

## CVPR-2023-RDD

- Local path: `datasets/paired/cvpr-2023-rdd`
- Paired document restoration dataset.
- Re-download/source details must follow upstream project or the existing `DATASETS.md` notes.

## Jung dataset

- Local path: `datasets/paired/jungs-dataset`
- Kaggle: <https://www.kaggle.com/datasets/xuhangc/jungs-dataset>

## Kligler dataset

- Local path: `datasets/paired/kliglers-dataset`
- Kaggle: <https://www.kaggle.com/datasets/xuhangc/kliglers-dataset>

## FSDSRD

- Local path: `datasets/paired/fsdsrd`
- Open-source benchmark/training data. Re-download from upstream project when needed.

## INV3D

- Local path: `datasets/paired/inv3d`
- Open-source/development benchmark data. Re-download from upstream project when needed.

## Local-only data: not open-source backup target

These remain excluded from GitHub and are handled separately:

- `datasets/paired/custom`
- `datasets/paired/custom_prepared`
- `datasets/controlpoints_custom`
- `datasets/magang`
- private training logs and generated outputs

## Restore rule

1. Clone GitHub source branch.
2. Read `README.md`, `AGENTS.md`, `DATASETS.md`, and `DATASET_DOWNLOAD.md`.
3. Download open-source datasets from upstream.
4. Restore custom datasets only from verified Google Drive backups.
5. Validate input/target pairs before training.

No credentials, tokens, `.env` files, or private connection data belong in this document.
      