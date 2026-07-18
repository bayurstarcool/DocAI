# ShadowDocument7K Dataset Download

## Problem
The OpenXLab CLI downloads files one by one, which is very slow (7000+ files).

## Solution: Download via Browser (Fastest)

### Step 1: Open Download Page
Go to: **https://openxlab.org.cn/datasets/lkljty/ShadowDocument7K**

### Step 2: Download ZIP
1. Click the **"Download"** button on the page
2. Select **"Download ZIP"** or **"Download All"**
3. The browser will download the ZIP file (~5 GB)

### Step 3: Upload to Server
Upload the downloaded ZIP file to:
```
/home/wahyu/docai/datasets/
```

### Step 4: Extract
```bash
cd /home/wahyu/docai
unzip datasets/ShadowDocument7K.zip -d datasets/
python3 prepare_dataset.py
```

## Alternative: Download Specific Files

If you only want to download specific files:

```bash
# Download only test set
openxlab dataset download \
  --dataset-repo "lkljty/ShadowDocument7K" \
  --source-path "/test" \
  --target-path "datasets/ShadowDocument7K"
```

## After Download

1. Verify the structure:
```bash
ls -la datasets/ShadowDocument7K/
ls -la datasets/ShadowDocument7K/train/input/ | head -5
ls -la datasets/ShadowDocument7K/train/target/ | head -5
```

2. Run preparation script:
```bash
python3 prepare_dataset.py
```

3. Start training:
```bash
cd docshadow_sd7k
python train.py
```

## References

- Dataset: https://openxlab.org.cn/datasets/lkljty/ShadowDocument7K
- Paper: https://arxiv.org/abs/2308.14221
- GitHub: https://github.com/CXH-Research/DocShadow-SD7K

## DocShadow-SD7K Pretrained Weights

Test Model mendukung pembanding pretrained dari https://github.com/CXH-Research/DocShadow-SD7K/releases.

Simpan file checkpoint ke:

```bash
checkpoints/docshadow/SD7K.pth
checkpoints/docshadow/Jung.pth
checkpoints/docshadow/Kligler.pth
```

Jika file ada, halaman Test Model otomatis menampilkan mode:

- Pretrained DocShadow SD7K
- Pretrained DocShadow Jung
- Pretrained DocShadow Kligler
