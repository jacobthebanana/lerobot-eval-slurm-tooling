#!/usr/bin/env bash
# run_eval_pusht_vqbet.sh — Submit closed-loop PushT eval using VQ-BeT
set -euo pipefail

N_EPISODES=50
BATCH_SIZE=10
SEED=1000
PARTITION="long"
TIMEOUT="24:00:00"
BACKEND="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-episodes) N_EPISODES="$2";       shift 2 ;;
        --batch-size) BATCH_SIZE="$2";       shift 2 ;;
        --seed)       SEED="$2";             shift 2 ;;
        --partition)  PARTITION="$2";        shift 2 ;;
        --timeout)    TIMEOUT="$2";          shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRATCH="${SCRATCH}"
VENV="${SCRATCH}/uv-venv/lerobot-libero-eval"
MODEL_DIR="${SCRATCH}/local-models/vqbet_pusht"
OUTPUT_DIR="${SCRATCH}/xvla-eval-output/$(date +%Y%m%d_%H%M%S)_vqbet_pusht"

mkdir -p "${SCRATCH}/xvla-eval-logs"

BATCH_SCRIPT=$(mktemp /tmp/vqbet_pusht_batch.XXXX.sh)
trap 'rm -f "$BATCH_SCRIPT"' EXIT

cat > "$BATCH_SCRIPT" << 'SCRIPTEOF'
#!/usr/bin/env bash
#SBATCH --job-name=vqbet-pusht-eval
#SBATCH --output=${SCRATCH}/xvla-eval-logs/vqbet-pusht-eval-%j.out
#SBATCH --error=${SCRATCH}/xvla-eval-logs/vqbet-pusht-eval-%j.err
#SBATCH --time=TIMEOUT_PLACEHOLDER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=PARTITION_PLACEHOLDER

set -euo pipefail

SCRATCH="${SCRATCH}"
VENV="${SCRATCH}/uv-venv/lerobot-libero-eval"
MODEL_DIR="${SCRATCH}/local-models/vqbet_pusht"
OUTPUT_DIR="OUTPUT_DIR_PLACEHOLDER"

export HF_HOME="${SCRATCH}/hf-cache"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${SCRATCH}/xdg-cache"

source "${VENV}/bin/activate"

echo "=== Environment ==="
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "N ep: N_EPISODES_PLACEHOLDER"
echo "Batch: BATCH_SIZE_PLACEHOLDER"

# Detect rendering backend
MUJOCO_BACKEND=""
if [ "BACKEND_PLACEHOLDER" = "osmesa" ]; then
    MUJOCO_BACKEND="osmesa"
elif [ "BACKEND_PLACEHOLDER" = "egl" ]; then
    MUJOCO_BACKEND="egl"
else
    if MUJOCO_GL=egl python -c "import gym_pusht" 2>/dev/null; then
        MUJOCO_BACKEND="egl"
    elif MUJOCO_GL=osmesa python -c "import gym_pusht" 2>/dev/null; then
        MUJOCO_BACKEND="osmesa"
    fi
fi
export MUJOCO_GL="${MUJOCO_BACKEND}"
export PYOPENGL_PLATFORM="${MUJOCO_BACKEND}"

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

echo "=== Eval complete ==="
ls -la OUTPUT_DIR_PLACEHOLDER/
SCRIPTEOF

sed -i \
    -e "s|TIMEOUT_PLACEHOLDER|${TIMEOUT}|g" \
    -e "s|PARTITION_PLACEHOLDER|${PARTITION}|g" \
    -e "s|N_EPISODES_PLACEHOLDER|${N_EPISODES}|g" \
    -e "s|BATCH_SIZE_PLACEHOLDER|${BATCH_SIZE}|g" \
    -e "s|SEED_PLACEHOLDER|${SEED}|g" \
    -e "s|BACKEND_PLACEHOLDER|${BACKEND}|g" \
    -e "s|OUTPUT_DIR_PLACEHOLDER|${OUTPUT_DIR}|g" \
    "$BATCH_SCRIPT"

echo "=== Submitting ==="
echo "  Model: vqbet_pusht -> PushT (local)"
echo "  N: ${N_EPISODES}, Batch: ${BATCH_SIZE}, Partition: ${PARTITION}"
sbatch --partition="${PARTITION}" "$BATCH_SCRIPT"
echo "=== Submitted ==="
