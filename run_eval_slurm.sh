#!/usr/bin/env bash
# run_eval_slurm.sh — Submit closed-loop LIBERO eval using lerobot-eval
#
# Usage:
#   bash run_eval_slurm.sh                                    # libero_spatial, 50 episodes
#   bash run_eval_slurm.sh --task libero_10                   # different suite
#   bash run_eval_slurm.sh --n-episodes 10 --batch-size 5     # quick test
#   bash run_eval_slurm.sh --task "libero_spatial,libero_object"  # multiple suites
set -euo pipefail

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
TASK_SUITE="libero_spatial"   # or libero_object, libero_goal, libero_10, libero_90
N_EPISODES=50
BATCH_SIZE=10
SEED=1000
PARTITION="long"              # use 'unkillable' for quick tests
TIMEOUT="24:00:00"
BACKEND="auto"                # auto, egl, osmesa — use osmesa to save GPU memory

# ------------------------------------------------------------------
# Parse args
# ------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)       TASK_SUITE="$2";       shift 2 ;;
        --n-episodes) N_EPISODES="$2";       shift 2 ;;
        --batch-size) BATCH_SIZE="$2";       shift 2 ;;
        --seed)       SEED="$2";             shift 2 ;;
        --partition)  PARTITION="$2";        shift 2 ;;
        --timeout)    TIMEOUT="$2";          shift 2 ;;
        --backend)    BACKEND="$2";          shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
SCRATCH="${SCRATCH}"
PROJECT_DIR="${HOME}/lerobot-eval-slurm-toolings"
VENV="${SCRATCH}/uv-venv/lerobot-libero-eval"
OUTPUT_DIR="${SCRATCH}/xvla-eval-output/$(date +%Y%m%d_%H%M%S)"

mkdir -p "${SCRATCH}/xvla-eval-logs"

BATCH_SCRIPT=$(mktemp /tmp/xvla_eval_batch.XXXX.sh)
trap 'rm -f "$BATCH_SCRIPT"' EXIT

cat > "$BATCH_SCRIPT" << 'SCRIPTEOF'
#!/usr/bin/env bash
#SBATCH --job-name=xvla-eval
#SBATCH --output=${SCRATCH}/xvla-eval-logs/xvla-eval-%j.out
#SBATCH --error=${SCRATCH}/xvla-eval-logs/xvla-eval-%j.err
#SBATCH --time=TIMEOUT_PLACEHOLDER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=PARTITION_PLACEHOLDER

set -euo pipefail

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
SCRATCH="${SCRATCH}"
VENV="${SCRATCH}/uv-venv/lerobot-libero-eval"

export HF_HOME="${SCRATCH}/hf-cache"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${SCRATCH}/xdg-cache"

# ------------------------------------------------------------------
# Ensure LIBERO config exists (prevents interactive prompt hang)
# ------------------------------------------------------------------
LIBERO_CONFIG_DIR="${HOME}/.libero"
LIBERO_CONFIG_FILE="${LIBERO_CONFIG_DIR}/config.yaml"
LIBERO_PKG_DIR="${VENV}/lib/python3.10/site-packages/libero/libero"
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
    echo "Created ${LIBERO_CONFIG_FILE}"
fi

source "${VENV}/bin/activate"

echo "=== Environment ==="
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Task suite: TASK_SUITE_PLACEHOLDER"
echo "N episodes: N_EPISODES_PLACEHOLDER"
echo "Batch size: BATCH_SIZE_PLACEHOLDER"
echo "Seed: SEED_PLACEHOLDER"
echo "Backend: BACKEND_PLACEHOLDER"

# Detect MuJoCo rendering backend
echo ""
echo "--- Detecting MuJoCo rendering backend ---"
MUJOCO_BACKEND=""

if [ "BACKEND_PLACEHOLDER" = "osmesa" ]; then
    # Force OSMesa to save GPU memory for the model
    if MUJOCO_GL=osmesa python -c "import robosuite" 2>/dev/null; then
        MUJOCO_BACKEND="osmesa"
        echo "Using OSMesa backend (forced — saves GPU VRAM)"
    else
        echo "FATAL: OSMesa backend not available."
        exit 1
    fi
elif [ "BACKEND_PLACEHOLDER" = "egl" ]; then
    if MUJOCO_GL=egl python -c "import robosuite" 2>/dev/null; then
        MUJOCO_BACKEND="egl"
        echo "Using EGL backend (forced — GPU-accelerated)"
    else
        echo "FATAL: EGL backend not available."
        exit 1
    fi
else
    # Auto: try EGL first, fall back to OSMesa
    if MUJOCO_GL=egl python -c "import robosuite" 2>/dev/null; then
        MUJOCO_BACKEND="egl"
        echo "Using EGL backend (GPU-accelerated)"
    elif MUJOCO_GL=osmesa python -c "import robosuite" 2>/dev/null; then
        MUJOCO_BACKEND="osmesa"
        echo "Using OSMesa backend (software rendering — slower but saves GPU VRAM)"
    else
        echo "FATAL: Neither EGL nor OSMesa backends work for robosuite."
        exit 1
    fi
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
echo "=== Starting closed-loop evaluation ==="

python -m lerobot.scripts.lerobot_eval \
    --policy.path=lerobot/xvla-libero \
    --env.type=libero \
    --env.task=TASK_SUITE_PLACEHOLDER \
    --env.obs_type=pixels_agent_pos \
    --env.control_mode=relative \
    --eval.n_episodes=N_EPISODES_PLACEHOLDER \
    --eval.batch_size=BATCH_SIZE_PLACEHOLDER \
    --policy.device=cuda \
    --policy.use_amp=false \
    --seed=SEED_PLACEHOLDER \
    --output_dir=OUTPUT_DIR_PLACEHOLDER

echo ""
echo "=== Eval complete ==="
echo "Results in: OUTPUT_DIR_PLACEHOLDER"
ls -la OUTPUT_DIR_PLACEHOLDER/
SCRIPTEOF

# Replace placeholders with actual values
sed -i \
    -e "s|TIMEOUT_PLACEHOLDER|${TIMEOUT}|g" \
    -e "s|PARTITION_PLACEHOLDER|${PARTITION}|g" \
    -e "s|TASK_SUITE_PLACEHOLDER|${TASK_SUITE}|g" \
    -e "s|N_EPISODES_PLACEHOLDER|${N_EPISODES}|g" \
    -e "s|BATCH_SIZE_PLACEHOLDER|${BATCH_SIZE}|g" \
    -e "s|SEED_PLACEHOLDER|${SEED}|g" \
    -e "s|BACKEND_PLACEHOLDER|${BACKEND}|g" \
    -e "s|OUTPUT_DIR_PLACEHOLDER|${OUTPUT_DIR}|g" \
    "$BATCH_SCRIPT"

echo "=== Submitting eval job ==="
echo "  Task suite:    ${TASK_SUITE}"
echo "  N episodes:    ${N_EPISODES}"
echo "  Batch size:    ${BATCH_SIZE}"
echo "  Partition:     ${PARTITION}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Python venv:   ${VENV}"
sbatch --partition="${PARTITION}" "$BATCH_SCRIPT"
echo "=== Submitted ==="
