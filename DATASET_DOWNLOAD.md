# ShadowDocument7K Dataset Download Guide

## Overview
- **Dataset**: SD7K (ShadowDocument7K)
- **Size**: 5.1 GB (compressed from OpenXLab) or 9 GB (compressed from OneDrive)
- **Resolution**: 2K (high-resolution)
- **Structure**: input/ (shadow) + target/ (clean)

## Download Methods

### Method 1: OpenXLab (Recommended for HPC)

1. Install OpenXLab CLI:
```bash
pip install openxlab
```

2. Login to OpenXLab:
```bash
openxlab login
```

3. Download dataset:
```bash
openxlab dataset download \
  --dataset-repo "lkljty/ShadowDocument7K" \
  --source-path "/" \
  --target-path "datasets/ShadowDocument7K"
```

OpenXLab dapat menambahkan folder `lkljty___ShadowDocument7K/`. Untuk training canonical, gunakan arsip `SD7K.zip` lalu ekstrak dengan `prepare_sd7k.py`.

### Method 2: OneDrive (Manual Download)

1. Go to: https://1drv.ms/u/s!Avp0JjwC1wv5jhx6TxHfjlL4BMuA?e=GFTcze
2. Download the compressed file (9 GB)
3. Extract to: `datasets/ShadowDocument7K/`

### Method 3: Kaggle (Alternative Datasets)

1. Kligler Dataset:
   - URL: https://www.kaggle.com/datasets/xuhangc/kliglers-dataset
   - Download: `kaggle datasets download -d xuhangc/kliglers-dataset`

2. Jung Dataset:
   - URL: https://www.kaggle.com/datasets/xuhangc/jungs-dataset
   - Download: `kaggle datasets download -d xuhangc/jungs-dataset`

## Expected Directory Structure

```
data/datasets/paired/ShadowDocument7K/
├── train/
│   ├── input/    # Shadow images
│   └── target/   # Clean images
└── test/
    ├── input/
    └── target/
```

## After Download

1. Extract archive ke layout training canonical:
```bash
python3 prepare_sd7k.py \
  --archive SD7K.zip \
  --output data/datasets/paired/ShadowDocument7K
```

2. Verify the structure:
```bash
ls -la data/datasets/paired/ShadowDocument7K/
ls -la data/datasets/paired/ShadowDocument7K/train/
ls -la data/datasets/paired/ShadowDocument7K/test/
```

3. Update config.yml in docshadow_sd7k/:
```yaml
TRAINING:
  TRAIN_DIR: '../data/datasets/paired/ShadowDocument7K/train/'
  VAL_DIR: '../data/datasets/paired/ShadowDocument7K/test/'
```

4. Start training:
```bash
cd docshadow_sd7k
python train.py
```

## References

- Paper: https://arxiv.org/abs/2308.14221
- GitHub: https://github.com/CXH-Research/DocShadow-SD7K
- OpenXLab: https://openxlab.org.cn/datasets/lkljty/ShadowDocument7K
