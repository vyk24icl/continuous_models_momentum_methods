#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/SGDM_venv/Scripts/python.exe}"

if [[ ! -f "${PYTHON_BIN}" ]]; then
    echo "Python executable not found:" >&2
    echo "  ${PYTHON_BIN}" >&2
    exit 1
fi

# CPU by default. This can be overridden externally if a compatible
# JAX GPU installation is available.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export PYTHONFAULTHANDLER=1
export JAX_TRACEBACK_FILTERING=off

ETA_DT_RATIO=10

PLOT_DIR="${SCRIPT_DIR}/plots/Computational efficiency/MNIST_MLP_small"
LOG_DIR="${SCRIPT_DIR}/logs/Computational efficiency/MNIST_MLP_small"
OUTPUT_LOG="${SCRIPT_DIR}/mnist_sme2_small.log"

mkdir -p "${PLOT_DIR}"
mkdir -p "${LOG_DIR}"

echo "Python executable: ${PYTHON_BIN}"
echo "Experiment: small MNIST M-SME-2 error"
echo "Integration ratio eta/dt: ${ETA_DT_RATIO}"
echo "Plot directory: ${PLOT_DIR}"
echo "TensorBoard directory: ${LOG_DIR}"

"${PYTHON_BIN}" -c \
    "import jax, flax; print('JAX:', jax.__version__); print('Flax:', flax.__version__); print('Devices:', jax.devices())"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/main_dist.py" \
    --sme2_only \
    --dataset mnist \
    --architecture nn \
    --hidden_widths 1024 \
    --activation softplus \
    --problem nn \
    --discrete_method sgdm \
    --mu 0.3 \
    --eta 0.005 \
    --sigma 0.02 \
    --num_steps 20 \
    --num_runs 20 \
    --eta_dt_ratio "${ETA_DT_RATIO}" \
    --brownian_tol_ratio 0.1 \
    --sde_batch_size 1 \
    --max_train_samples 1024 \
    --max_test_samples 128 \
    --seed 0 \
    --plot_dir "${PLOT_DIR}" \
    --log_dir "${LOG_DIR}" \
    2>&1 | tee "${OUTPUT_LOG}"