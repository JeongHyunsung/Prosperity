#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SIM_PY="${SIM_PY:-simulator.py}"            # <-- change to your entry script path
ROOT_OUT_DIR="${ROOT_OUT_DIR:-../output_dse_cache}"  # <-- parent folder for all runs

# Base args (keep consistent across runs)
BASE_ARGS=(
  --type Prosperity
  --tile_size_M 256
  --tile_size_K 16
)

# DSE params
MIN_TH=1
MAX_THS=(3 4 5)
POLICIES=(max_nnz lru fifo random)
M_MULTS=(1 2 4)

mkdir -p "${ROOT_OUT_DIR}"
echo "[DSE] ROOT_OUT_DIR=${ROOT_OUT_DIR}"
echo "[DSE] SIM_PY=${SIM_PY}"
echo

total=0
fail=0

for max_th in "${MAX_THS[@]}"; do
  for policy in "${POLICIES[@]}"; do
    for mm in "${M_MULTS[@]}"; do
      total=$((total+1))

      tag="min${MIN_TH}_max${max_th}_pol${policy}_mm${mm}"
      run_dir="${ROOT_OUT_DIR}/${tag}"
      log_file="${run_dir}/run.log.txt"

      mkdir -p "${run_dir}"

      echo "=== [${total}] ${tag} ==="
      echo "  output_dir: ${run_dir}"
      echo "  log:        ${log_file}"

      set +e
      "${PYTHON_BIN}" "${SIM_PY}" \
        "${BASE_ARGS[@]}" \
        --use_global_cache \
        --min_th "${MIN_TH}" \
        --max_th "${max_th}" \
        --cache_policy "${policy}" \
        --m_multiple "${mm}" \
        --output_dir "${run_dir}" \
        > "${log_file}" 2>&1
      rc=$?
      set -e

      if [[ $rc -ne 0 ]]; then
        fail=$((fail+1))
        echo "[FAIL] rc=${rc} ${tag}" | tee -a "${log_file}"
      else
        echo "[OK]   ${tag}" | tee -a "${log_file}"
      fi

      echo
    done
  done
done

echo "=============================="
echo "[DSE DONE] total=${total}, fail=${fail}"
echo "All results are under: ${ROOT_OUT_DIR}"
