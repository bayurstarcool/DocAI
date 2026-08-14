#!/usr/bin/env bash
set -Eeuo pipefail
# Run after downloading files from Google Drive.
need(){ [ -e "$1" ] || { echo "MISSING $1" >&2; exit 2; }; }
parts=(docai_model_backup_20260813.tar.part-*)
[ "${#parts[@]}" -eq 13 ] || { echo "EXPECTED 13 model parts, found ${#parts[@]}" >&2; exit 3; }
mapfile -t parts < <(printf '%s\n' "${parts[@]}" | sort)
total=0
for p in "${parts[@]}"; do total=$((total + $(stat -c %s "$p"))); done
[ "$total" -eq 412559360 ] || { echo "MODEL PART BYTES MISMATCH: $total" >&2; exit 4; }
cat "${parts[@]}" > docai_model_backup_20260813.tar
sha256sum docai_model_backup_20260813.tar
printf '%s\n' 'MODEL ARCHIVE PART COUNT/BYTES PASS'
md5sum controlpoints_custom.tar paired_custom_prepared.tar
printf '%s\n' 'Compare MD5 with RESTORE.md before extraction.'
      