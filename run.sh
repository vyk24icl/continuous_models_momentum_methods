#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Keep the complete ${...} expression on one line.
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/NN_venv/Scripts/python.exe}"

if [[ ! -f "${PYTHON_BIN}" ]]; then
    echo "Python executable not found:" >&2
    echo "  ${PYTHON_BIN}" >&2
    exit 1
fi

# Real-valued CPU experiment.
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONFAULTHANDLER=1
export JAX_TRACEBACK_FILTERING=off

ETA_DT_RATIO=100

PLOT_DIR="${SCRIPT_DIR}/plots/fig10_gdm"
LOG_DIR="${SCRIPT_DIR}/logs/sgdm_real_30_steps_ratio_${ETA_DT_RATIO}"
OUTPUT_LOG="${SCRIPT_DIR}/sgdm_real_30_steps_ratio_${ETA_DT_RATIO}.log"

mkdir -p "${PLOT_DIR}"
mkdir -p "${LOG_DIR}"

EXTRA_ARGS=()

# Optional initial-Hessian-only diagnostic:
#
#   HESSIAN_ONLY=1 ./run.sh
if [[ "${HESSIAN_ONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--hessian_diagnostic_only)
fi

echo "Python executable: ${PYTHON_BIN}"
echo "Integration ratio: ${ETA_DT_RATIO}"

"${PYTHON_BIN}" -c "import jax, flax; print('JAX:', jax.__version__); print('Flax:', flax.__version__); print('Devices:', jax.devices())"
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/main_dist.py" \
    --fig13_gdm \
    --figure13_only \
    --mu 0.1 \
    --eta 0.3 \
    --num_steps 80 \
    --architecture nn \
    --hidden_widths 10 10 10 10 \
    --activation tanh \
    --dataset iris \
    --discrete_method gdm \
    --problem nn \
    --seed 0 \
    --plot_dir "${PLOT_DIR}" \
    --log_dir "${LOG_DIR}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_LOG}"


    # --recurrence_local_errors \
    # --momentum_dal_fig19 \
    # --dal_fixed_etas 0.1 0.2 0.3 0.4 0.5 0.6 \
    # --dal_target_ratio 1.0 \
    # --dal_min_eta 1e-8 \
    # --dal_max_eta 0.5 \

    # --max_train_samples 1000 \
    # --max_test_samples 200 \

    # --fig13_gdm \
    # --figure13_only \
    # --fig13_start_step 1 \
    # --fig13_end_step 100 \
    # --fig13_flow_substeps 200 \


    # --complex_pf \
    # --num_runs 1 \
    # --sigma 0.00 \
    # --eta_dt_ratio "${ETA_DT_RATIO}" \
    # --brownian_tol_ratio 0.1 \
    # --sde_batch_size 1 \

    # --quad_h11 7.0 \
    # --quad_h12 1.0 \
    # --quad_h22 1.0 \
    # --quad_target_1 1.0 \
    # --quad_target_2 1.0 \
    # --quad_init_1 1.0 \
    # --quad_init_2 2.0 \