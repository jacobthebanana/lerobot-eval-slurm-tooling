#!/usr/bin/env bash
# run_experiments.sh — Submit parallel experiments testing X-VLA hypotheses
#
# Each experiment varies one hypothesis dimension:
#   1. Denoising steps: 5, 10 (baseline), 25, 50, 100
#   2. Ensemble samples: 1 (baseline), 3, 5
#   3. Sample timing: predict every 5, 10, 20 steps
#
# Usage:
#   bash run_experiments.sh                        # all experiments
#   bash run_experiments.sh --quick                 # fast subset for testing
#   bash run_experiments.sh --partition long        # use long partition
set -euo pipefail

PARTITION="long"
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PARTITION="$2"; shift 2 ;;
        --quick) QUICK_MODE=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SCRATCH="${SCRATCH}"
PROJECT_DIR="${HOME}/lerobot-eval-slurm-toolings"
VENV="${SCRATCH}/uv-venv/lerobot-demo"

# Experiment definitions: "exp_name python_args"
# Format: denoising_steps num_samples sample_every
if $QUICK_MODE; then
    EXPERIMENTS=(
        "ds10_ns1  10 1 10"
        "ds50_ns1  50 1 10"
    )
    MAX_EPS=2
else
    EXPERIMENTS=(
        # Vary denoising steps
        "ds5_ns1    5  1 10"
        "ds10_ns1  10  1 10"
        "ds25_ns1  25  1 10"
        "ds50_ns1  50  1 10"
        "ds100_ns1 100 1 10"
        # Vary ensemble samples
        "ds10_ns3  10  3 10"
        "ds10_ns5  10  5 10"
        # Vary sample timing
        "ds10_ns1_se5  10 1 5"
        "ds10_ns1_se20 10 1 20"
        # High denoising + ensemble
        "ds50_ns3  50  3 10"
    )
    MAX_EPS=5
fi

echo "=== Submitting X-VLA experiments ==="
echo "Partition: $PARTITION"
echo "Max episodes per experiment: $MAX_EPS"
echo "Number of experiments: ${#EXPERIMENTS[@]}"
echo ""

JOB_IDS=()

for exp in "${EXPERIMENTS[@]}"; do
    read -r NAME DS NS SE <<< "$exp"

    BATCH_SCRIPT=$(mktemp /tmp/xvla_exp_${NAME}_XXXXXX.sh)

    cat > "$BATCH_SCRIPT" << SCRIPTEOF
#!/usr/bin/env bash
#SBATCH --job-name=xvla-${NAME}
#SBATCH --output=${PROJECT_DIR}/logs/xvla-${NAME}-%j.out
#SBATCH --error=${PROJECT_DIR}/logs/xvla-${NAME}-%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
export HF_HOME="\${SCRATCH}/hf-cache"
export PYTHONUNBUFFERED=1
source "${VENV}/bin/activate"
cd "${PROJECT_DIR}"

echo "=== Experiment: ${NAME} ==="
echo "denoising_steps=${DS}, num_samples=${NS}, sample_every=${SE}"
echo "Python: \$(which python)"
echo "CUDA: \$(python -c 'import torch; print(torch.cuda.is_available())')"

mkdir -p logs output/experiments

exec python -u experiment_inference.py \
    --denoising-steps ${DS} \
    --num-samples ${NS} \
    --sample-every ${SE} \
    --max-episodes ${MAX_EPS} \
    --output-dir output/experiments
SCRIPTEOF

    echo "  Submitting ${NAME}: ds=${DS}, ns=${NS}, se=${SE}"
    JOB_ID=$(sbatch --partition="${PARTITION}" --parsable "$BATCH_SCRIPT")
    JOB_IDS+=("$JOB_ID")
    echo "    Job ID: $JOB_ID"
done

echo ""
echo "=== Submitted ${#JOB_IDS[@]} jobs ==="
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with: squeue -j ${JOB_IDS[*]}"
echo "Or: watch -n 5 'squeue -j ${JOB_IDS[*]}'"
