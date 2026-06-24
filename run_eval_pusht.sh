#!/usr/bin/env bash
# run_eval_pusht.sh — Submit closed-loop PushT eval using diffusion_pusht
#
# Usage:
#   bash run_eval_pusht.sh                              # 50 episodes
#   bash run_eval_pusht.sh --n-episodes 10 --batch-size 5  # quick test
set -euo pipefail

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
N_EPISODES=50
BATCH_SIZE=10
SEED=1000
PARTITION="long"
TIMEOUT="24:00:00"
BACKEND="auto"

# ------------------------------------------------------------------
# Parse args
# ------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
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
MODEL_DIR="${SCRATCH}/local-models/diffusion_pusht"
OUTPUT_DIR="${SCRATCH}/xvla-eval-output/$(date +%Y%m%d_%H%M%S)_pusht"

mkdir -p "${SCRATCH}/xvla-eval-logs"

BATCH_SCRIPT=$(mktemp /tmp/pusht_eval_batch.XXXX.sh)
trap 'rm -f "$BATCH_SCRIPT"' EXIT

cat > "$BATCH_SCRIPT" << 'SCRIPTEOF'
#!/usr/bin/env bash
#SBATCH --job-name=diff-pusht-eval
#SBATCH --output=${SCRATCH}/xvla-eval-logs/diff-pusht-eval-%j.out
#SBATCH --error=${SCRATCH}/xvla-eval-logs/diff-pusht-eval-%j.err
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
MODEL_DIR="${SCRATCH}/local-models/diffusion_pusht"

export HF_HOME="${SCRATCH}/hf-cache"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${SCRATCH}/xdg-cache"

source "${VENV}/bin/activate"

echo "=== Environment ==="
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "N episodes: N_EPISODES_PLACEHOLDER"
echo "Batch size: BATCH_SIZE_PLACEHOLDER"
echo "Seed: SEED_PLACEHOLDER"
echo "Backend: BACKEND_PLACEHOLDER"

# Detect MuJoCo rendering backend
echo ""
echo "--- Detecting MuJoCo rendering backend ---"
MUJOCO_BACKEND=""

if [ "BACKEND_PLACEHOLDER" = "osmesa" ]; then
    if MUJOCO_GL=osmesa python -c "import gym_pusht" 2>/dev/null; then
        MUJOCO_BACKEND="osmesa"
        echo "Using OSMesa backend (forced)"
    else
        echo "FATAL: OSMesa backend not available."
        exit 1
    fi
elif [ "BACKEND_PLACEHOLDER" = "egl" ]; then
    if MUJOCO_GL=egl python -c "import gym_pusht" 2>/dev/null; then
        MUJOCO_BACKEND="egl"
        echo "Using EGL backend (forced)"
    else
        echo "FATAL: EGL backend not available."
        exit 1
    fi
else
    # Auto
    if MUJOCO_GL=egl python -c "import gym_pusht" 2>/dev/null; then
        MUJOCO_BACKEND="egl"
        echo "Using EGL backend (GPU-accelerated)"
    elif MUJOCO_GL=osmesa python -c "import gym_pusht" 2>/dev/null; then
        MUJOCO_BACKEND="osmesa"
        echo "Using OSMesa backend (software rendering)"
    else
        echo "FATAL: Neither EGL nor OSMesa backends work."
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
    --policy.path="${MODEL_DIR}" \
    --env.type=pusht \
    --env.task=PushT-v0 \
    --env.obs_type=pixels_agent_pos \
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
    -e "s|N_EPISODES_PLACEHOLDER|${N_EPISODES}|g" \
    -e "s|BATCH_SIZE_PLACEHOLDER|${BATCH_SIZE}|g" \
    -e "s|SEED_PLACEHOLDER|${SEED}|g" \
    -e "s|BACKEND_PLACEHOLDER|${BACKEND}|g" \
    -e "s|OUTPUT_DIR_PLACEHOLDER|${OUTPUT_DIR}|g" \
    "$BATCH_SCRIPT"

echo "=== Submitting eval job ==="
echo "  Model:       diffusion_pusht (local)"
echo "  Env type:    pusht"
echo "  Task:        PushT-v0"
echo "  N episodes:  ${N_EPISODES}"
echo "  Batch size:  ${BATCH_SIZE}"
echo "  Partition:   ${PARTITION}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Python venv: ${VENV}"
sbatch --partition="${PARTITION}" "$BATCH_SCRIPT"
echo "=== Submitted ==="
