#!/usr/bin/env bash
# run_eval_google_robot_slurm.sh — Submit SimplerEnv eval for lerobot/xvla-google-robot
#
# Usage:
#   bash run_eval_google_robot_slurm.sh                                                   # default task
#   bash run_eval_google_robot_slurm.sh --task google_robot_pick_coke_can,google_robot_move_near
#   bash run_eval_google_robot_slurm.sh --n-episodes 10 --max-steps 100
#   bash run_eval_google_robot_slurm.sh --partition unkillable --timeout 4:00:00
#
# Note: SimplerEnv must be installed first via:
#   bash install_simplerenv.sh
set -euo pipefail

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
TASK="google_robot_pick_coke_can"
N_EPISODES=50
MAX_STEPS=200
SEED=1000
PARTITION="long"
TIMEOUT="24:00:00"

# ------------------------------------------------------------------
# Parse args
# ------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)       TASK="$2";          shift 2 ;;
        --n-episodes) N_EPISODES="$2";     shift 2 ;;
        --max-steps)  MAX_STEPS="$2";      shift 2 ;;
        --seed)       SEED="$2";           shift 2 ;;
        --partition)  PARTITION="$2";      shift 2 ;;
        --timeout)    TIMEOUT="$2";        shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
SCRATCH="${SCRATCH}"
PROJECT_DIR="${HOME}/lerobot-eval-slurm-toolings"
VENV="${SCRATCH}/uv-venv/simplerenv"
OUTPUT_DIR="${SCRATCH}/xvla-eval-output/google_robot/$(date +%Y%m%d_%H%M%S)"

mkdir -p "${SCRATCH}/xvla-eval-logs"

BATCH_SCRIPT=$(mktemp /tmp/xvla_google_eval_batch.XXXX.sh)
trap 'rm -f "$BATCH_SCRIPT"' EXIT

cat > "$BATCH_SCRIPT" << 'SCRIPTEOF'
#!/usr/bin/env bash
#SBATCH --job-name=xvla-google-eval
#SBATCH --output=${SCRATCH}/xvla-eval-logs/xvla-google-eval-%j.out
#SBATCH --error=${SCRATCH}/xvla-eval-logs/xvla-google-eval-%j.err
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
VENV="${SCRATCH}/uv-venv/simplerenv"
PROJECT_DIR="${HOME}/lerobot-eval-slurm-toolings"

export HF_HOME="${SCRATCH}/hf-cache"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${SCRATCH}/xdg-cache"
export MUJOCO_GL=egl
# SimplerEnv/SAPIEN requires libvulkan.so.1
export LD_LIBRARY_PATH="${SCRATCH}/local-libs/vulkan/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

source "${VENV}/bin/activate"

echo "=== Environment ==="
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "SimlperEnv available: $(python -c 'import simpler_env; print(simpler_env.__file__)' 2>&1)"
echo "Task: TASK_PLACEHOLDER"
echo "N episodes: N_EPISODES_PLACEHOLDER"
echo "Max steps: MAX_STEPS_PLACEHOLDER"
echo "Seed: SEED_PLACEHOLDER"
echo ""

echo "=== Starting evaluation ==="

python "${PROJECT_DIR}/eval_xvla_google_robot.py" \
    --task=TASK_PLACEHOLDER \
    --n-episodes=N_EPISODES_PLACEHOLDER \
    --max-steps=MAX_STEPS_PLACEHOLDER \
    --device=cuda \
    --seed=SEED_PLACEHOLDER \
    --output-dir=OUTPUT_DIR_PLACEHOLDER

echo ""
echo "=== Eval complete ==="
echo "Results in: OUTPUT_DIR_PLACEHOLDER"
ls -la OUTPUT_DIR_PLACEHOLDER/ 2>/dev/null || echo "(dir may be empty if error)"
SCRIPTEOF

sed -i \
    -e "s|TIMEOUT_PLACEHOLDER|${TIMEOUT}|g" \
    -e "s|PARTITION_PLACEHOLDER|${PARTITION}|g" \
    -e "s|TASK_PLACEHOLDER|${TASK}|g" \
    -e "s|N_EPISODES_PLACEHOLDER|${N_EPISODES}|g" \
    -e "s|MAX_STEPS_PLACEHOLDER|${MAX_STEPS}|g" \
    -e "s|SEED_PLACEHOLDER|${SEED}|g" \
    -e "s|OUTPUT_DIR_PLACEHOLDER|${OUTPUT_DIR}|g" \
    "$BATCH_SCRIPT"

echo "=== Submitting eval job ==="
echo "  Task:          ${TASK}"
echo "  N episodes:    ${N_EPISODES}"
echo "  Max steps:     ${MAX_STEPS}"
echo "  Partition:     ${PARTITION}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Python venv:   ${VENV}"
sbatch --partition="${PARTITION}" "$BATCH_SCRIPT"
echo "=== Submitted ==="
