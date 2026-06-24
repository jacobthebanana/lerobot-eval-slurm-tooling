#!/usr/bin/env bash
# run_slurm.sh — Submit GPU inference job to SLURM
#
# Usage:
#   bash run_slurm.sh                                    # all tasks, unkillable partition
#   bash run_slurm.sh --task-filter "on_the_stove"       # specific task
#   bash run_slurm.sh --max-episodes 2 --max-episodes-per-task 1  # test run
#   bash run_slurm.sh --partition long                    # use long partition
set -euo pipefail

PARTITION="unkillable"
PY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition)
            PARTITION="$2"; shift 2 ;;
        *)
            PY_ARGS+=("$1"); shift ;;
    esac
done

SCRATCH="${SCRATCH}"
PROJECT_DIR="${HOME}/lerobot-eval-slurm-toolings"

BATCH_SCRIPT=$(mktemp /tmp/xvla_libero_batch.XXXXXX.sh)
trap 'rm -f "$BATCH_SCRIPT"' EXIT

cat > "$BATCH_SCRIPT" << SCRIPTEOF
#!/usr/bin/env bash
#SBATCH --job-name=xvla-libero
#SBATCH --output=${SCRATCH}/xvla-inference-logs/xvla-libero-%j.out
#SBATCH --error=${SCRATCH}/xvla-inference-logs/xvla-libero-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

SCRATCH="\${SCRATCH}"
VENV="\${SCRATCH}/uv-venv/lerobot-demo"

export HF_HOME="\${SCRATCH}/hf-cache"
export PYTHONUNBUFFERED=1

mkdir -p "\${SCRATCH}/xvla-inference-logs"
mkdir -p "\${SCRATCH}/xvla-inference-output/raw_data"

source "\${VENV}/bin/activate"

echo "=== Environment ==="
echo "Python: \$(which python)"
echo "CUDA: \$(python -c 'import torch; print(torch.cuda.is_available())')"

cd "${PROJECT_DIR}"
exec python -u run_inference.py ${PY_ARGS[@]}
SCRIPTEOF

echo "Submitting to partition: ${PARTITION}"
echo "Python args: ${PY_ARGS[@]}"
sbatch --partition="${PARTITION}" "$BATCH_SCRIPT"
