#!/usr/bin/env bash
# Run the standard sine, damped_SHM, and airline configurations and build aggregate plots.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
PYTHON_BIN="${PYTHON:-python}"

cd "${ROOT_DIR}"
mkdir -p "${RESULTS_DIR}"

datasets=(sine damped_shm airline)

for dataset in "${datasets[@]}"; do
  echo "=== Dataset: ${dataset} ==="
  run_args=()
  snapshots=""
  configs=()

  case "${dataset}" in
    sine)
      snapshots="1,15,30,100"
      configs=(
        "sine_lstm.json:LSTM"
        "sine_qlstm.json:QLSTM"
        "sine_qlstm_photonic.json:Photonic QLSTM"
      )
      ;;
    damped_shm)
      snapshots="1,15,30,100"
      configs=(
        "damped_shm_lstm.json:LSTM"
        "damped_shm_qlstm.json:QLSTM"
        "damped_shm_qlstm_photonic.json:Photonic QLSTM"
      )
      ;;
    airline)
      snapshots="1,25,50,100"
      configs=(
        "airline_lstm.csv.json:LSTM"
        "airline_qlstm.csv.json:QLSTM"
        "airline_qlstm_photonic.csv.json:Photonic QLSTM"
      )
      ;;
    *)
      echo "Unknown dataset key: ${dataset}" >&2
      exit 1
      ;;
  esac

  for entry in "${configs[@]}"; do
    IFS=':' read -r cfg label <<< "${entry}"
    cfg_path="configs/${cfg}"
    if [[ ! -f "${cfg_path}" ]]; then
      echo "Missing config: ${cfg_path}" >&2
      exit 1
    fi

    echo "--- Running ${cfg_path}"
    log_file="$(mktemp)"
    if ! "${PYTHON_BIN}" implementation.py --config "${cfg_path}" | tee "${log_file}"; then
      echo "Training failed for ${cfg_path}" >&2
      cat "${log_file}" >&2
      rm -f "${log_file}"
      exit 1
    fi

    run_dir=$(find outdir -maxdepth 1 -type d -name 'run_*' -print | sort -r | head -n1)
    rm -f "${log_file}"
    if [[ -z "${run_dir}" ]]; then
      echo "Could not determine run directory for ${cfg_path}" >&2
      exit 1
    fi

    display_label="${label//_/ }"
    echo "    -> ${display_label} run: ${run_dir}"
    run_args+=("${run_dir}:${display_label}")
  done

  out_path="${RESULTS_DIR}/${dataset}_aggregate.png"
  echo "--- Rendering aggregate plot: ${out_path}"
  "${PYTHON_BIN}" -m utils.aggregate_plots \
    --runs "${run_args[@]}" \
    --epochs "${snapshots}" \
    --out "${out_path}" \
    --width 4 --height 3

  loss_path="${RESULTS_DIR}/${dataset}_loss_comparison.png"
  echo "--- Rendering loss comparison: ${loss_path}"
  "${PYTHON_BIN}" -m utils.compare_losses \
    --runs "${run_args[@]}" \
    --metric test \
    --out "${loss_path}"

  echo "=== Done: ${dataset} ==="
  echo
done
