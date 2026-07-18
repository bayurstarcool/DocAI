# 📊 DocAI Dataset Management Guide

## Overview

DocAI supports multiple datasets for document shadow removal and enhancement training. This guide explains how to manage and organize datasets.

## Available Datasets

### DocShadow Format (input/target pairs)

| Dataset | Pairs | Description | Source |
|---------|-------|-------------|--------|
| **DocShadow-Prepared** | 500 | Shadow removal pairs | Generated |
| **DocShadow-Color** | 500 | Color enhancement pairs | Image-Colorization |
| **DocShadow-Enhance** | 500 | Document enhancement pairs | Generated |
| **DocShadow-ColorCast** | 200 | Color cast correction pairs | Generated |

### Downloaded Datasets

| Dataset | Images | Description | Source |
|---------|--------|-------------|--------|
| **Image-Colorization** | 6,000 | Diverse scene images | HuggingFace |
| **Document_Enhancement** | 500 | Document images | Generated |
| **Document_Color_Correction** | 400 | Color correction pairs | Generated |
| **Document_Deskew** | 200 | Deskew pairs | Generated |

### Prepared Training Datasets

| Dataset | Pairs | Description |
|---------|-------|-------------|
| **Shadow Removal** | 500 | Shadow removal training pairs |
| **Enhancement** | 500 | Document enhancement pairs |

## Dataset Structure

### DocShadow Format (Required for DocShadow-SD7K)

```
datasets/SD7K/
├── input/          # Shadow/degraded images
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── target/         # Clean images
    ├── 0001.png
    ├── 0002.png
    └── ...
```

### Alternative Format

```
data/shadow_removal/
├── clean/          # Original images
└── shadowed/       # Shadow-applied images
```

## Downloading Datasets

### 1. SD7K Dataset (Recommended)

**Size**: ~9GB (compressed) or ~100GB (original)

```bash
# From OpenXLab (Recommended for HPC)
pip install openxlab
openxlab dataset download --repo lkljty/ShadowDocument7K --local ./datasets/SD7K

# From OneDrive
# Compressed: https://1drv.ms/u/s!Avp0JjwC1wv5jhx6TxHfjlL4BMuA?e=GFTcze
# Original: https://1drv.ms/f/s!Avp0JjwC1wv5a4DsxiA0-swHw9A?e=GWLPl3
```

### 2. Kligler Dataset

```bash
# From Kaggle
pip install kaggle
kaggle datasets download -d xuhangc/kliglers-dataset
unzip kliglers-dataset.zip -d ./datasets/Kligler
```

### 3. Jung Dataset

```bash
# From Kaggle
kaggle datasets download -d xuhangc/jungs-dataset
unzip jungs-dataset.zip -d ./datasets/Jung
```

## Managing Datasets

### List All Datasets

```python
from manage_docshadow_datasets import DocShadowDatasetManager

manager = DocShadowDatasetManager()
datasets = manager.list_datasets()

for d in datasets:
    print(f"{d['name']}: {d['input_count']} input, {d['target_count']} target")
```

### Validate Dataset

```python
validation = manager.validate_dataset("SD7K")
print(f"Valid pairs: {validation['valid_pairs']}")
```

### Prepare Dataset from Images

```python
manager.prepare_from_existing_images(
    "MyDataset",
    "/path/to/images",
    create_pairs=True  # Create synthetic pairs
)
```

### Export for Training

```python
manager.export_for_training(
    "SD7K",
    "/home/wahyu/docai/data/SD7K_training"
)
```

## API Endpoints

### List Datasets
```bash
GET /api/datasets
```

### Response Format
```json
{
  "datasets": [
    {
      "name": "DocShadow-Prepared",
      "path": "/home/wahyu/docai/datasets/DocShadow-Prepared",
      "count": 500,
      "type": "docshadow",
      "input_count": 500,
      "target_count": 500,
      "format": "input/target"
    }
  ]
}
```

## Training with Datasets

### Using DocShadow-SD7K

```python
from DocShadow-SD7K.data import get_training_data

# Configure
train_dir = "/home/wahyu/docai/datasets/SD7K"
val_dir = "/home/wahyu/docai/datasets/Kligler"

# Load data
train_dataset = get_training_data(train_dir, {'w': 512, 'h': 512})
val_dataset = get_validation_data(val_dir, {'w': 512, 'h': 512, 'ori': True})
```

### Using Our Models

```python
from backend.datasets.document_dataset import ShadowRemovalDataset

dataset = ShadowRemovalDataset(
    "/home/wahyu/docai/datasets/DocShadow-Prepared",
    size=512
)
```

## Pre-trained Weights

Download pre-trained weights from DocShadow-SD7K:

```bash
# SD7K weights
wget https://github.com/CXH-Research/DocShadow-SD7K/releases/download/Weights/SD7K.pth

# Jung weights
wget https://github.com/CXH-Research/DocShadow-SD7K/releases/download/Weights/Jung.pth

# Kligler weights
wget https://github.com/CXH-Research/DocShadow-SD7K/releases/download/Weights/Kligler.pth
```

Place weights in: `/home/wahyu/docai/checkpoints/docshadow/`

## Dataset Statistics

```bash
# Run dataset manager
python3 manage_docshadow_datasets.py

# Or check via API
curl http://localhost:8000/api/datasets | python3 -m json.tool
```

## Tips

1. **Use DocShadow Format**: Organize datasets with `input/` and `target/` folders
2. **Validate Datasets**: Always validate before training
3. **Start Small**: Begin with smaller datasets (500 pairs) before scaling up
4. **Use Pre-trained Weights**: Start training from pre-trained weights for better results
5. **Monitor GPU Usage**: Ensure sufficient VRAM for large datasets

## References

- [DocShadow-SD7K Paper](https://arxiv.org/abs/2308.14221)
- [DocShadow-SD7K GitHub](https://github.com/CXH-Research/DocShadow-SD7K)
- [SD7K Dataset](https://openxlab.org.cn/datasets/lkljty/ShadowDocument7K)
