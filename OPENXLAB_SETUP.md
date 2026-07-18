# OpenXLab Setup Guide

## Problem
The token you provided appears to be an **Access Key (AK)** only.

OpenXLab requires both **AK (Access Key)** and **SK (Secret Key)** to download datasets.

## Solution

### Step 1: Get your SK (Secret Key)

1. Go to: https://sso.openxlab.org.cn/usercenter?tab=secret
2. Login with your account
3. Find your **Secret Key (SK)**
4. Copy the SK

### Step 2: Configure OpenXLab

```bash
# Option 1: Use openxlab login (interactive)
openxlab login -r

# When prompted:
# - Paste your Access Key (AK)
# - Paste your Secret Key (SK): [your SK here]
```

### Step 3: Download Dataset

```bash
cd /home/wahyu/docai

openxlab dataset download \
  --dataset-repo "lkljty/ShadowDocument7K" \
  --source-path "/" \
  --target-path "datasets/ShadowDocument7K"
```

## Alternative: Manual Download

If you can't get the SK, download manually:

1. Go to: https://openxlab.org.cn/datasets/lkljty/ShadowDocument7K
2. Click "Download" button
3. Download the dataset files
4. Extract to: `datasets/ShadowDocument7K/`

## Your Current Config

```ini
[openxlab]
ak = <OPENXLAB_ACCESS_KEY_ID>
sk = [MISSING - need your SK]
```

## After Download

Once the dataset is downloaded:

```bash
# Prepare dataset
python3 prepare_dataset.py

# Start training
cd docshadow_sd7k
python train.py
```
