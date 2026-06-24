#!/usr/bin/env bash
# run_eval_vqbet_pusht_v2.sh — Submit closed-loop PushT eval using VQ-BeT
#
# Usage:
#   bash run_eval_vqbet_pusht_v2.sh
#   bash run_eval_vqbet_pusht_v2.sh --n-episodes 10 --batch-size 5
set -euo pipefail

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
N_EPISODES=50
BATCH_SIZE=5
SEED=1000
PARTITION="long"
TIMEOUT="24:00:00"
VENV_PATH="${SCRATCH}/uv-venv/lerobot-new"
POLICY_PATH="lerobot/vqbet_pusht"
SCRATCH="${SCRATCH}"
OUTPUT_BASE="${SCRATCH}/vqbet-eval-output-v2"
LOG_DIR="${SCRATCH}/vqbet-eval-logs-v2"
PROJECT_DIR="${HOME}/lerobot-eval-slurm-toolings"

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
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

timestamp=$(date +%Y%m%d_%H%M%S)
output_dir="${OUTPUT_BASE}/${timestamp}"
mkdir -p "$output_dir"

BATCH_SCRIPT=$(mktemp /tmp/vqbet_eval_XXXX.sh)
trap 'rm -f "$BATCH_SCRIPT"' EXIT

cat > "$BATCH_SCRIPT" << 'SCRIPTEOF'
#!/usr/bin/env bash
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
# Environment
# ------------------------------------------------------------------
SCRATCH="${SCRATCH}"
VENV="VENV_PATH_PLACEHOLDER"

export HF_HOME="${SCRATCH}/hf-cache"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${SCRATCH}/xdg-cache"

source "${VENV}/bin/activate"

echo "=== Environment ==="
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "LeRobot version: $(python -c 'import lerobot; print(lerobot.__version__)')"
echo "N episodes: N_EPISODES_PLACEHOLDER"
echo "Batch size: BATCH_SIZE_PLACEHOLDER"
echo "Seed: SEED_PLACEHOLDER"

# ------------------------------------------------------------------
# Preprocess config: download and strip mlp_hidden_dim from vqbet config
# (the current LeRobot codebase doesn't have this field, but the hub model does)
# ------------------------------------------------------------------
echo ""
echo "--- Preprocessing VQ-BeT config ---"
PREPROCESSED_DIR="${SCRATCH}/.vqbet_preprocessed_config"
mkdir -p "${PREPROCESSED_DIR}"

python -c "
import json, os
from huggingface_hub import hf_hub_download

config_file = hf_hub_download(
    repo_id='lerobot/vqbet_pusht',
    filename='config.json'
)
with open(config_file) as f:
    config = json.load(f)

# Remove fields not recognized by current VQBeTConfig
removed = config.pop('mlp_hidden_dim', None)
if removed is not None:
    print(f'Removed mlp_hidden_dim={removed} from config')

# Also download model weights and other files
import shutil
from huggingface_hub import list_repo_files, hf_hub_download

files = list_repo_files('lerobot/vqbet_pusht')
for fname in files:
    if fname.startswith('.'):
        continue
    local_path = os.path.join('${PREPROCESSED_DIR}', fname)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    dl_path = hf_hub_download(repo_id='lerobot/vqbet_pusht', filename=fname)
    shutil.copy2(dl_path, local_path)
    print(f'Copied {fname}')

# Write modified config
with open(os.path.join('${PREPROCESSED_DIR}', 'config.json'), 'w') as f:
    json.dump(config, f)

print('Preprocessed config saved to ${PREPROCESSED_DIR}')
"

echo "--- Preprocessing complete ---"

echo ""
echo "=== Starting closed-loop evaluation ==="

python -m lerobot.scripts.lerobot_eval \
    --policy.path="${PREPROCESSED_DIR}" \
    --env.type=pusht \
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

job_name="vqbet-pusht"
sed -i \
    -e "s|JOB_NAME_PLACEHOLDER|${job_name}|g" \
    -e "s|LOG_DIR_PLACEHOLDER|${LOG_DIR}|g" \
    -e "s|TIMEOUT_PLACEHOLDER|${TIMEOUT}|g" \
    -e "s|PARTITION_PLACEHOLDER|${PARTITION}|g" \
    -e "s|N_EPISODES_PLACEHOLDER|${N_EPISODES}|g" \
    -e "s|BATCH_SIZE_PLACEHOLDER|${BATCH_SIZE}|g" \
    -e "s|SEED_PLACEHOLDER|${SEED}|g" \
    -e "s|VENV_PATH_PLACEHOLDER|${VENV_PATH}|g" \
    -e "s|POLICY_PATH_PLACEHOLDER|${POLICY_PATH}|g" \
    -e "s|OUTPUT_DIR_PLACEHOLDER|${output_dir}|g" \
    "$BATCH_SCRIPT"

echo "=== Submitting eval job ==="
echo "  Policy:        ${POLICY_PATH} (with preprocessed config)"
echo "  N episodes:    ${N_EPISODES}"
echo "  Batch size:    ${BATCH_SIZE}"
echo "  Partition:     ${PARTITION}"
echo "  Output dir:    ${output_dir}"
echo "  Python venv:   ${VENV_PATH}"
sbatch --partition="${PARTITION}" "$BATCH_SCRIPT"
echo "=== Submitted ==="
