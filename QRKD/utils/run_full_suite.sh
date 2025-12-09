#!/usr/bin/env bash
set -euo pipefail

# Run the full QRKD MNIST suite:
# - Teacher (10 epochs)
# - Scratch student
# - KD student
# - RKD student
# - QRKD (simple)
# - QRKD (Merlin backend)
# - QRKD (Qiskit backend)
# Generates a summary table in results/report.txt

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
OUTDIR="${PROJECT_DIR}/outdir"
RESULTS_DIR="${PROJECT_DIR}/results"

# inherit python binary from environment, or use default
PYTHON_BIN="python"

mkdir -p "${OUTDIR}" "${RESULTS_DIR}"

# Defaults (override with --dataset / --epochs)
DATASET="mnist"
CEPOCHS="10"
EPOCHS="10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2;;
    --epochs)
      EPOCHS="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

run_exp() {
  local cfg="$1"; shift
  local extra_args=()
  if [[ $# -gt 0 ]]; then
    extra_args=("$@")
  fi
  local tmp_log
  tmp_log="$(mktemp)"
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    "${PYTHON_BIN}" "${ROOT_DIR}/implementation.py" --project QRKD --config "${cfg}" "${extra_args[@]}" | tee "${tmp_log}" >&2
  else
    "${PYTHON_BIN}" "${ROOT_DIR}/implementation.py" --project QRKD --config "${cfg}" | tee "${tmp_log}" >&2
  fi
  local status=${PIPESTATUS[0]}
  if [[ ${status} -ne 0 ]]; then
    echo "ERROR: training failed (status ${status}). See log ${tmp_log}" >&2
    exit 1
  fi

  # Parse run log from the most recent run directory
  local recent_log
  recent_log="$(ls -1dt "${OUTDIR}"/run_*/run.log 2>/dev/null | head -1)"
  if [[ -z "${recent_log}" ]]; then
    echo "ERROR: could not locate run.log after training. See log ${tmp_log}" >&2
    exit 1
  fi
  local reported
  reported="$(grep -F 'Artifacts in:' "${recent_log}" | tail -1 | sed 's/.*Artifacts in: //')"
  if [[ -z "${reported}" ]]; then
    echo "ERROR: failed to parse run dir from ${recent_log}. See log ${tmp_log}" >&2
    exit 1
  fi
  rm -f "${tmp_log}"
  if [[ "${reported}" != /* ]]; then
    reported="${PROJECT_DIR}/${reported}"
  fi
  echo "${reported}"
}

echo "Running teacher (${EPOCHS} epochs)..."
TEACHER_RUN="$(run_exp "${PROJECT_DIR}/configs/teacher_${DATASET}_${CEPOCHS}epochs.json" --epochs "${EPOCHS}")"
echo "Teacher run: ${TEACHER_RUN}"

echo "Running scratch student..."
SCRATCH_RUN="$(run_exp "${PROJECT_DIR}/configs/student_scratch_${DATASET}_${CEPOCHS}epochs.json" --epochs "${EPOCHS}")"
echo "Scratch run: ${SCRATCH_RUN}"

echo "Running KD student..."
KD_RUN="$(run_exp "${PROJECT_DIR}/configs/student_kd_${DATASET}_${CEPOCHS}epochs.json" --teacher-path "${TEACHER_RUN}/teacher.pt" --epochs "${EPOCHS}")"
echo "KD run: ${KD_RUN}"

echo "Running RKD student..."
RKD_RUN="$(run_exp "${PROJECT_DIR}/configs/student_rkd_${DATASET}_${CEPOCHS}epochs.json" --teacher-path "${TEACHER_RUN}/teacher.pt" --epochs "${EPOCHS}")"
echo "RKD run: ${RKD_RUN}"

echo "Running QRKD (simple backend)..."
QRKD_SIMPLE_RUN="$(run_exp "${PROJECT_DIR}/configs/student_qrkd_simple_${DATASET}_${CEPOCHS}epochs.json" --teacher-path "${TEACHER_RUN}/teacher.pt" --qkernel-backend simple --epochs "${EPOCHS}")"
echo "QRKD simple run: ${QRKD_SIMPLE_RUN}"

echo "Running QRKD (Merlin backend)..."
QRKD_MERLIN_RUN="$(run_exp "${PROJECT_DIR}/configs/student_qrkd_merlin_${DATASET}_${CEPOCHS}epochs.json" --teacher-path "${TEACHER_RUN}/teacher.pt" --epochs "${EPOCHS}")"
echo "QRKD merlin run: ${QRKD_MERLIN_RUN}"

echo "Running QRKD (Qiskit backend)..."
#QRKD_QISKIT_RUN="$(run_exp "${PROJECT_DIR}/configs/student_qrkd_qiskit_${DATASET}_${CEPOCHS}epochs.json" --teacher-path "${TEACHER_RUN}/teacher.pt" --epochs "${EPOCHS}")"
#echo "QRKD qiskit run: ${QRKD_QISKIT_RUN}"

SUFFIX="${DATASET}_${EPOCHS}epochs"
REPORT_PATH="${RESULTS_DIR}/report_${SUFFIX}.txt"
echo "Building report at ${REPORT_PATH}"
"${PYTHON_BIN}" "${PROJECT_DIR}/utils/report.py" \
  --teacher "${TEACHER_RUN}" \
  --scratch "${SCRATCH_RUN}" \
  --kd "${KD_RUN}" \
  --rkd "${RKD_RUN}" \
  --qrkd-simple "${QRKD_SIMPLE_RUN}" \
  --qrkd-merlin "${QRKD_MERLIN_RUN}" \
  > "${REPORT_PATH}"

echo "Building accuracy plot and combined histories..."
"${PYTHON_BIN}" "${PROJECT_DIR}/utils/plot_history.py" \
  --teacher "${TEACHER_RUN}" \
  --scratch "${SCRATCH_RUN}" \
  --kd "${KD_RUN}" \
  --rkd "${RKD_RUN}" \
  --qrkd-simple "${QRKD_SIMPLE_RUN}" \
  --qrkd-merlin "${QRKD_MERLIN_RUN}" \
  --out-json "${RESULTS_DIR}/history_combined_${SUFFIX}.json" \
  --out-plot "${RESULTS_DIR}/accuracy_plot_${SUFFIX}.png"

echo "Done. Report saved to ${REPORT_PATH}"
