# DocAI Restore Runbook

Status date: 2026-08-14
Source repository: `github.com/bayurstarcool/DocAI`

## 1. Restore source

```bash
git clone --branch backup/server-state-20260814 git@github.com:bayurstarcool/DocAI.git docai
cd docai
```

Do not copy `.env`, tokens, SSH keys, or rclone config from any backup.
Create fresh credentials on target host.

## 2. Restore verified model archive

Primary model archive:

```text
gdrive:/DocAI-Backup/docai_model_backup_20260813_parts/
```

Expected parts:

```text
13 parts: aa through am
Total: 412559360 bytes
```

Download all parts into one directory. Verify count and total bytes before joining:

```bash
find . -maxdepth 1 -type f -name 'docai_model_backup_20260813.tar.part-*' | sort
cat docai_model_backup_20260813.tar.part-* > docai_model_backup_20260813.tar
sha256sum docai_model_backup_20260813.tar
```

Only extract after expected count and archive checksum from backup manifest match.
Preserve existing production checkpoint before replacement. Never overwrite Base blindly.

## 3. Restore Mixed ONNX artifacts

```text
gdrive:/DocAI-Backup/mixed_best_onnx_20260813/files/
```

Expected files:

```text
docres_appearance_mixed_best_iter1500.onnx
docres_appearance_mixed_best_iter1500.json
mixed_lift_v2.py
mixed_lift_v3.py
README.txt
SHA256SUMS.txt
```

Verify SHA256SUMS before runtime use. ONNX is Mixed-only; it does not include V5 or LIFT.

## 4. Restore exact custom datasets

Verified exact archives:

```text
gdrive:/DocAI-Backup/datasets/controlpoints_custom.tar
gdrive:/DocAI-Backup/datasets/paired_custom_prepared.tar
```

Verify MD5 against the backup record before extraction:

```text
controlpoints_custom.tar: abbbb126f49c6a04d4d9f68c6a628c63
paired_custom_prepared.tar: 41b9d4b8255a10aeb560e5bdecb41619
```

## 5. Re-download open-source datasets

Use `OPEN_SOURCE_DATASETS.md`, `DATASETS.md`, and `DATASET_DOWNLOAD.md`.
Do not waste Drive space copying open-source datasets again.
Validate paired input/target names after download.

## 6. Restore runtime

```bash
cd /path/to/docai
export PYTHONPATH=$PWD
/home/wahyu/miniconda3/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Current production route was:

```text
docres_appearance_mixed_v5_lift12
```

Validate Base, Mixed, V5, and full route separately. Do not label Mixed ONNX as Mixed+V5+LIFT.

## Quarantine rule

Do not restore any object under these paths as a complete archive:

```text
gdrive:/DocAI-Backup/server_20260814/docres_core/
gdrive:/DocAI-Backup/custom_selected_20260814/custom_final_20260804/
gdrive:/DocAI-Backup/custom_selected_20260814_v2/custom_review_20260804/
```

These are partial/incomplete transfers. They remain preserved as evidence only.

## Credential rule

Credentials were intentionally excluded. Recreate SSH, GitHub, Google Drive, API, and application secrets manually on the new host.
      