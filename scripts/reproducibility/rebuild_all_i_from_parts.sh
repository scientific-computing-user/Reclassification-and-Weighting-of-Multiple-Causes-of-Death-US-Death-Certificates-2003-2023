#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      echo "Usage: $0 --work-dir <processed_snapshot_dir>" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${WORK_DIR}" ]]; then
  echo "[error] Missing --work-dir argument" >&2
  echo "Usage: $0 --work-dir <processed_snapshot_dir>" >&2
  exit 2
fi

ALL_I_FILE="${WORK_DIR}/all_I=converted_with_original.csv"
TMP_FILE="${ALL_I_FILE}.tmp"
PARTS_DIR="${WORK_DIR}/all_I_parts"

if [[ -s "${ALL_I_FILE}" ]]; then
  echo "[prep] all_I source already present: ${ALL_I_FILE}"
  exit 0
fi

shopt -s nullglob
parts=( "${PARTS_DIR}/all_I=converted_with_original.csv.gz.part."* )
shopt -u nullglob

if (( ${#parts[@]} == 0 )); then
  echo "[prep] No split all_I parts found at: ${PARTS_DIR}"
  exit 0
fi

echo "[prep] Rebuilding all_I source from ${#parts[@]} split gzip parts"
echo "[prep] Target file: ${ALL_I_FILE}"

if cat "${parts[@]}" | gzip -dc > "${TMP_FILE}"; then
  mv "${TMP_FILE}" "${ALL_I_FILE}"
  echo "[prep] Rebuild complete ($(stat -f%z "${ALL_I_FILE}") bytes)"
else
  rm -f "${TMP_FILE}"
  echo "[error] Failed to rebuild ${ALL_I_FILE} from split parts" >&2
  exit 1
fi
