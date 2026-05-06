#!/usr/bin/env bash
set -uo pipefail

SCRIPT=${1:-profile_coalesce_compare.py}
OUT_DIR=${2:-ncu_reports}
mkdir -p "${OUT_DIR}"

COMMON_APP_ARGS=(
  --n-elements $((512 * 1024 * 1024))
  --num-warps 2
  --warmup 1
  --iters 1
  --profile-api
)

COMMON_NCU_ARGS=(
  --target-processes all
  --profile-from-start off
  --set full
  --force-overwrite
)

for MODE in bs512 bs256; do
  echo "========== Profiling ${MODE} =========="

  ncu "${COMMON_NCU_ARGS[@]}" \
    --export "${OUT_DIR}/${MODE}_full" \
    --log-file "${OUT_DIR}/${MODE}_ncu.log" \
    python "${SCRIPT}" \
    --mode "${MODE}" \
    "${COMMON_APP_ARGS[@]}"

  if [[ $? -ne 0 ]]; then
    echo "[ERROR] NCU profiling failed for ${MODE}"
    echo "Check: ${OUT_DIR}/${MODE}_ncu.log"
    continue
  fi

  if [[ ! -f "${OUT_DIR}/${MODE}_full.ncu-rep" ]]; then
    echo "[ERROR] Missing report: ${OUT_DIR}/${MODE}_full.ncu-rep"
    echo "Probably no kernels were profiled for ${MODE}"
    echo "Check: ${OUT_DIR}/${MODE}_ncu.log"
    continue
  fi

  echo "========== Exporting details report for ${MODE} =========="
  ncu \
    --import "${OUT_DIR}/${MODE}_full.ncu-rep" \
    --page details \
    --print-details all \
    --print-metric-name label-name \
    > "${OUT_DIR}/${MODE}_details.txt"

  echo "========== Exporting raw CSV report for ${MODE} =========="
  ncu \
    --import "${OUT_DIR}/${MODE}_full.ncu-rep" \
    --page raw \
    --csv \
    --print-units base \
    --print-metric-name label-name \
    > "${OUT_DIR}/${MODE}_raw.csv"
done

echo "========== Done =========="
ls -lh "${OUT_DIR}"