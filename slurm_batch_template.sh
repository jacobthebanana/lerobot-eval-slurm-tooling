#!/usr/bin/env bash
# slurm_batch_template.sh — Reusable template for SLURM eval jobs with NFS-safe venv extraction
#
# This template packages the venv extraction, LIBERO config, MuJoCo backend detection,
# and environment setup into a single template. Placeholders (JOB_NAME_PLACEHOLDER, etc.)
# are substituted by the launcher script at submit time.
#
# Usage:
#   cp this file, fill placeholders, then sbatch.
#
# Placeholders:
#   JOB_NAME_PLACEHOLDER      — SLURM job name
#   LOG_DIR_PLACEHOLDER       — directory for stdout/stderr logs
#   TIMEOUT_PLACEHOLDER       — wall time (e.g. 24:00:00)
#   PARTITION_PLACEHOLDER     — SLURM partition
#   VENV_TARBALL_PLACEHOLDER  — path to the venv tarball on shared storage
#   VENV_NAME_PLACEHOLDER     — directory name inside the tarball
#   PYTHON_VERSION_PLACEHOLDER — python version string for site-packages path (e.g. python3.10)
#   EXTRA_ENV_SETUP_PLACEHOLDER — extra environment setup (e.g. LD_LIBRARY_PATH for vulkan)
#   EVAL_COMMAND_PLACEHOLDER   — the actual eval command to run
#   OUTPUT_DIR_PLACEHOLDER     — directory for eval results
#
# Examples:
#   # LIBERO job
#   sed "s|VENV_TARBALL_PLACEHOLDER|/path/to/lerobot-libero-eval-v2.tar.gz|g; ..." template.sh | sbatch
#
#   # SimplerEnv job
#   sed "s|VENV_TARBALL_PLACEHOLDER|/path/to/simplerenv.tar.gz|g; \
#        s|EXTRA_ENV_SETUP_PLACEHOLDER|export LD_LIBRARY_PATH=...|g; ..." template.sh | sbatch
#
#SBATCH --job-name=JOB_NAME_PLACEHOLDER
#SBATCH --output=LOG_DIR_PLACEHOLDER/JOB_NAME_PLACEHOLDER-%j.out
#SBATCH --error=LOG_DIR_PLACEHOLDER/JOB_NAME_PLACEHOLDER-%j.err
#SBATCH --time=TIMEOUT_PLACEHOLDER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=PARTITION_PLACEHOLDER

set -euo pipefail

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
SCRATCH="${SCRATCH}"
LOCAL_TMP="/tmp/${SLURM_JOB_ID}"
mkdir -p "$LOCAL_TMP"

export HF_HOME="${SCRATCH}/hf-cache"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${SCRATCH}/xdg-cache"

# ------------------------------------------------------------------
# Extract venv from tarball to local disk (avoids NFS stale-file issues)
# ------------------------------------------------------------------
echo "=== Extracting venv to local tmp ==="
tar -xzf "VENV_TARBALL_PLACEHOLDER" -C "$LOCAL_TMP"
VENV="$LOCAL_TMP/VENV_NAME_PLACEHOLDER"
echo "  Venv extracted to: ${VENV}"

# ------------------------------------------------------------------
# Extra environment setup (vulkan, etc.)
# ------------------------------------------------------------------
EXTRA_ENV_SETUP_PLACEHOLDER

# ------------------------------------------------------------------
# LIBERO config setup (only for LIBERO tasks)
# ------------------------------------------------------------------
LIBERO_CONFIG_DIR="${HOME}/.libero"
LIBERO_CONFIG_FILE="${LIBERO_CONFIG_DIR}/config.yaml"
LIBERO_PKG_DIR="${VENV}/lib/PYTHON_VERSION_PLACEHOLDER/site-packages/libero/libero"
LIBERO_DATASETS_DIR="${SCRATCH}/libero-datasets"

if [ ! -f "${LIBERO_CONFIG_FILE}" ]; then
    echo "--- Creating LIBERO config ---"
    mkdir -p "${LIBERO_CONFIG_DIR}" "${LIBERO_DATASETS_DIR}"
    cat > "${LIBERO_CONFIG_FILE}" << EOF
benchmark_root: ${LIBERO_PKG_DIR}
bddl_files: ${LIBERO_PKG_DIR}/bddl_files
init_states: ${LIBERO_PKG_DIR}/init_files
datasets: ${LIBERO_DATASETS_DIR}
assets: ${LIBERO_PKG_DIR}/assets
EOF
    echo "  Created ${LIBERO_CONFIG_FILE}"
fi

source "${VENV}/bin/activate"

echo "=== Environment ==="
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ------------------------------------------------------------------
# Detect MuJoCo rendering backend
# ------------------------------------------------------------------
echo ""
echo "--- Detecting MuJoCo rendering backend ---"
MUJOCO_BACKEND=""

if MUJOCO_GL=egl python -c "import robosuite" 2>/dev/null; then
    MUJOCO_BACKEND="egl"
    echo "  Using EGL backend (GPU-accelerated)"
elif MUJOCO_GL=osmesa python -c "import robosuite" 2>/dev/null; then
    MUJOCO_BACKEND="osmesa"
    echo "  Using OSMesa backend (software rendering)"
else
    echo "  WARNING: Neither EGL nor OSMesa backends work for robosuite."
    echo "  Will try EGL anyway — may work depending on mujoco version."
    MUJOCO_BACKEND="egl"
fi

export MUJOCO_GL="${MUJOCO_BACKEND}"
export PYOPENGL_PLATFORM="${MUJOCO_BACKEND}"

# NVIDIA EGL ICD for GPU nodes
if [ "${MUJOCO_BACKEND}" = "egl" ]; then
    for egl_dir in \
        /usr/share/glvnd/egl_vendor.d \
        /etc/glvnd/egl_vendor.d \
        /usr/lib/x86_64-linux-gnu/GL ; do
        if [ -d "$egl_dir" ]; then
            export __GLX_VENDOR_LIBRARY_NAME="nvidia"
            break
        fi
    done
    if [ -e /dev/dri/renderD128 ]; then
        export EGL_DEVICE_ID=0
    fi
fi

echo ""
echo "=== Starting evaluation ==="

EVAL_COMMAND_PLACEHOLDER

echo ""
echo "=== Eval complete ==="
echo "Results in: OUTPUT_DIR_PLACEHOLDER"

# Cleanup local tmp
rm -rf "$LOCAL_TMP"
echo "  Cleaned up ${LOCAL_TMP}"
