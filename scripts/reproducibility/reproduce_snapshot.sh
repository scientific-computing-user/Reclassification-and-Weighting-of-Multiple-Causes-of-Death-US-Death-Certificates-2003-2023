#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
PIPE_DIR="${PIPE_DIR:-${ROOT_DIR}/scripts/pipeline}"
WORK_DIR="${DATA_DIR}/processed_snapshot"
OUT_DIR="${1:-${ROOT_DIR}/results/reproduced}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REBUILD_ALL_I_SCRIPT="${ROOT_DIR}/scripts/reproducibility/rebuild_all_i_from_parts.sh"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[error] Missing required file: ${path}" >&2
    exit 1
  fi
}

echo "[info] Root: ${ROOT_DIR}"
echo "[info] Data dir: ${DATA_DIR}"
echo "[info] Pipeline dir: ${PIPE_DIR}"
echo "[info] Work dir: ${WORK_DIR}"
echo "[info] Output dir: ${OUT_DIR}"

require_file "${PIPE_DIR}/create_record_entity_table_v2.py"
require_file "${PIPE_DIR}/create_disease_weight_tables_v4.py"
require_file "${PIPE_DIR}/mortality_plot_popscaled_v8.py"
require_file "${PIPE_DIR}/trend_analysis_Table20.py"
require_file "${PIPE_DIR}/analyze_rADS_W2A+++_v2.py"
require_file "${PIPE_DIR}/analyze_questions_Q1-Q38_v2.py"
require_file "${PIPE_DIR}/ucod_preference.py"
require_file "${WORK_DIR}/I_converted_with_original-f3,5,9,ag=sort=uniq-c=sort-rn.H.csv"
require_file "${WORK_DIR}/I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv"
require_file "${WORK_DIR}/I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv"
require_file "${WORK_DIR}/month_idx_agg_all_schemes.csv"
require_file "${DATA_DIR}/reference/USA_Population5.txt"
require_file "${DATA_DIR}/reference/icd10_to_DL_lookup_v4.csv"
require_file "${REBUILD_ALL_I_SCRIPT}"

mkdir -p "${OUT_DIR}/tables"
mkdir -p "${OUT_DIR}/figures"
mkdir -p "${OUT_DIR}/supplement"
mkdir -p "${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}/validation"

echo "[prep] Rebuild split all_I source file when parts are available"
"${REBUILD_ALL_I_SCRIPT}" --work-dir "${WORK_DIR}" \
  > "${OUT_DIR}/logs/00_rebuild_all_i.log" 2>&1

echo "[step 1/8] Record vs Entity tables"
"${PYTHON_BIN}" "${PIPE_DIR}/create_record_entity_table_v2.py" \
  "${WORK_DIR}/I_converted_with_original-f3,5,9,ag=sort=uniq-c=sort-rn.H.csv" \
  --years 2003-2023 \
  --label "USA 2003-2023" \
  --output "${OUT_DIR}/tables/Record_Entity_Comparison_2003-2023.xlsx" \
  > "${OUT_DIR}/logs/01_record_entity.log" 2>&1

echo "[step 2/8] Disease weight tables"
"${PYTHON_BIN}" "${PIPE_DIR}/create_disease_weight_tables_v4.py" \
  "${WORK_DIR}/I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv" \
  --all-years 2003-2023 \
  --pandemic-years 2020-2023 \
  --output "${OUT_DIR}/tables/Disease_Weights" \
  > "${OUT_DIR}/logs/02_disease_weight_tables.log" 2>&1

echo "[step 3/8] Trend analysis table"
(
  cd "${WORK_DIR}"
  "${PYTHON_BIN}" "${PIPE_DIR}/trend_analysis_Table20.py" month_idx_agg_all_schemes.csv "${DATA_DIR}/reference/USA_Population5.txt"
) > "${OUT_DIR}/logs/03_trend_analysis.log" 2>&1
cp "${WORK_DIR}/Table20_trend_analysis.xlsx" "${OUT_DIR}/tables/Table20_trend_analysis.xlsx"

echo "[step 4/8] Figure assets (population-scaled)"
"${PYTHON_BIN}" "${PIPE_DIR}/mortality_plot_popscaled_v8.py" \
  --agg_csv "${WORK_DIR}/month_idx_agg_all_schemes.csv" \
  --pop_txt "${DATA_DIR}/reference/USA_Population5.txt" \
  --out_dir "${OUT_DIR}/figures" \
  --all_scheme \
  > "${OUT_DIR}/logs/04_mortality_plots.log" 2>&1

echo "[step 5/8] rADS/rUDS W2A comparison tables"
"${PYTHON_BIN}" "${PIPE_DIR}/analyze_rADS_W2A+++_v2.py" \
  "${WORK_DIR}/I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv" \
  --years 2003-2023 \
  --output "${OUT_DIR}/supplement/rADS_W2A_analysis_2003-2023" \
  > "${OUT_DIR}/logs/05_rads_w2a_2003_2023.log" 2>&1
"${PYTHON_BIN}" "${PIPE_DIR}/analyze_rADS_W2A+++_v2.py" \
  "${WORK_DIR}/I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv" \
  --years 2020-2023 \
  --output "${OUT_DIR}/supplement/rADS_W2A_analysis_2020-2023" \
  > "${OUT_DIR}/logs/05_rads_w2a_2020_2023.log" 2>&1

echo "[step 6/8] Q1-Q38 summary table"
gzip -k -f "${WORK_DIR}/I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv"
"${PYTHON_BIN}" "${PIPE_DIR}/analyze_questions_Q1-Q38_v2.py" \
  "${WORK_DIR}/I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv.gz" \
  "${OUT_DIR}/supplement/questions_Q1-Q38.tsv" \
  > "${OUT_DIR}/logs/06_questions_q1_q38.log" 2>&1

if [[ -f "${WORK_DIR}/I_converted_with_original-f18=sort=uniq-c=sort-rn.H.csv" ]]; then
  echo "[step 7/8] UCOD preference table"
  "${PYTHON_BIN}" "${PIPE_DIR}/ucod_preference.py" \
    "${WORK_DIR}/I_converted_with_original-f18=sort=uniq-c=sort-rn.H.csv" \
    0 \
    > "${OUT_DIR}/supplement/ucod_preference.csv" \
    2> "${OUT_DIR}/logs/07_ucod_preference.log"
else
  echo "[step 7/8] UCOD preference table (skipped: large source file not included)"
fi

if [[ -s "${WORK_DIR}/all_I=converted_with_original.csv" ]]; then
  echo "[step 8/8] Claim validation report"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/reproducibility/validate_claims.py" \
    --all-i "${WORK_DIR}/all_I=converted_with_original.csv" \
    --lookup "${DATA_DIR}/reference/icd10_to_DL_lookup_v4.csv" \
    --entity-column sel_code \
    --output-json "${OUT_DIR}/validation/claim_metrics.json" \
    --output-md "${OUT_DIR}/validation/claim_check.md" \
    > "${OUT_DIR}/logs/08_claim_validation.log" 2>&1
else
  echo "[step 8/8] Claim validation report (skipped: large all_I source file not included)"
fi

echo "[done] Reproduction complete."
echo "[done] Outputs are in: ${OUT_DIR}"
