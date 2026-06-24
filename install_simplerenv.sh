#!/usr/bin/env bash
# install_simplerenv.sh — Install SimplerEnv for X-VLA Google Robot evaluation
#
# SimplerEnv (https://github.com/simpler-env/SimplerEnv) is a SAPIEN-based
# simulation environment that supports Google Robot tasks.
#
# Requirements:
#   - CUDA >= 11.8 (for SAPIEN GPU rendering)
#   - NVIDIA GPU (RTX series recommended)
#   - Python 3.10
#
# Usage:
#   bash install_simplerenv.sh [--venv PATH]
#
# Options:
#   --venv PATH   Path to existing venv (default: creates new at $SCRATCH/uv-venv/simplerenv)

set -euo pipefail

SCRATCH="${SCRATCH}"
DEFAULT_VENV="${SCRATCH}/uv-venv/simplerenv"
VENV="${DEFAULT_VENV}"
if [ "${1:-}" = "--venv" ]; then
    VENV="$2"
fi

echo "=== Installing SimplerEnv ==="
echo "  VENV:  ${VENV}"
echo "  Date:  $(date)"
echo ""

# ------------------------------------------------------------------
# 1. Create / activate venv
# ------------------------------------------------------------------
if [ ! -d "${VENV}" ]; then
    echo "--- Creating virtual environment ---"
    uv venv "${VENV}" --python 3.10
fi

source "${VENV}/bin/activate"

# ------------------------------------------------------------------
# 2. Install base ML dependencies
# ------------------------------------------------------------------
echo "--- Installing core ML packages ---"
uv pip install --upgrade pip
uv pip install "wheel"  # ensure wheel is available

# numpy < 2.0 is required for SimplerEnv IK
uv pip install "numpy<2.0" torch torchvision

# ------------------------------------------------------------------
# 3. Install lerobot with xvla support
# ------------------------------------------------------------------
echo "--- Installing LeRobot with XVLA support ---"
uv pip install "lerobot[xvla]"

# ------------------------------------------------------------------
# 4. Clone and install SimplerEnv
# ------------------------------------------------------------------
echo "--- Cloning SimplerEnv ---"
SIMPLER_DIR="${SCRATCH}/simpler-env"

if [ ! -d "${SIMPLER_DIR}" ]; then
    git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules "${SIMPLER_DIR}"
else
    echo "SimplerEnv already cloned at ${SIMPLER_DIR}; updating..."
    cd "${SIMPLER_DIR}"
    git pull --recurse-submodules
fi

# Install ManiSkill2_real2sim
echo "--- Installing ManiSkill2_real2sim ---"
cd "${SIMPLER_DIR}/ManiSkill2_real2sim"
uv pip install -e .

# Install SimplerEnv
echo "--- Installing SimplerEnv ---"
cd "${SIMPLER_DIR}"
uv pip install -e .

# ------------------------------------------------------------------
# 5. Ensure libvulkan.so.1 is available (needed by SAPIEN)
# ------------------------------------------------------------------
echo "--- Installing libvulkan (needed by SAPIEN) ---"
VULKAN_DIR="${SCRATCH}/local-libs/vulkan"
if [ ! -f "${VULKAN_DIR}/usr/lib/x86_64-linux-gnu/libvulkan.so.1" ]; then
    mkdir -p "${VULKAN_DIR}"
    TMP_DEB=$(mktemp /tmp/libvulkan1.XXXX.deb)
    apt-get download -o Dir::Cache=/tmp libvulkan1 2>/dev/null || \
        (cd /tmp && apt-get download libvulkan1 2>/dev/null)
    # Find the downloaded deb
    DEB_FILE=$(ls /tmp/libvulkan1*.deb 2>/dev/null | head -1)
    if [ -n "${DEB_FILE}" ]; then
        dpkg-deb -x "${DEB_FILE}" "${VULKAN_DIR}"
        rm -f "${DEB_FILE}"
        echo "  libvulkan installed at ${VULKAN_DIR}"
    else
        echo "  WARNING: Could not download libvulkan1. You may need to install it manually."
        echo "  See: https://github.com/simpler-env/SimplerEnv for instructions."
    fi
else
    echo "  libvulkan already present at ${VULKAN_DIR}"
fi

echo ""
echo "=== Installation complete ==="
echo "  To activate: source ${VENV}/bin/activate"
echo "  To test:     export LD_LIBRARY_PATH=\"${VULKAN_DIR}/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH}\""
echo "  Then:        python -c \"import simpler_env; env = simpler_env.make('google_robot_pick_coke_can'); print('SimplerEnv ready:', env)\""
echo "  Or use the script: bash ${SCRATCH}/20260623-lerobot-xvla/run_eval_google_robot_slurm.sh"
echo ""
