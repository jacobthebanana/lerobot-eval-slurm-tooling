#!/usr/bin/env bash
# test_state_fix.sh — Compare old 8D state vs new 20D state on a single task
set -euo pipefail

SCRATCH="${SCRATCH}"
PROJECT_DIR="${HOME}/lerobot-eval-slurm-toolings"
VENV="${SCRATCH}/uv-venv/lerobot-demo"

BATCH_SCRIPT=$(mktemp /tmp/xvla_state_fix_test.XXXXXX.sh)
trap 'rm -f "$BATCH_SCRIPT"' EXIT

cat > "$BATCH_SCRIPT" << SCRIPTEOF
#!/usr/bin/env bash
#SBATCH --job-name=xvla-fix-test
#SBATCH --output=${PROJECT_DIR}/logs/xvla-fix-test-%j.out
#SBATCH --error=${PROJECT_DIR}/logs/xvla-fix-test-%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
export HF_HOME="${SCRATCH}/hf-cache"
export PYTHONUNBUFFERED=1
source "${VENV}/bin/activate"
cd "${PROJECT_DIR}"

LOGDIR="${SCRATCH}/xvla-fix-logs"
mkdir -p "${LOGDIR}"

echo "=== Testing 20D state (official pipeline match) ==="
mkdir -p ${SCRATCH}/xvla-fix-comparison/20d

python -u run_inference.py \
    --task-filter "on_the_stove" \
    --max-episodes 1 \
    --predict-every 10 \
    --output-dir ${SCRATCH}/xvla-fix-comparison/20d \
    2>&1 | tee "${LOGDIR}/xvla-fix-20d.log"

echo ""
echo "=== Testing 8D state (legacy, for comparison) ==="
mkdir -p ${SCRATCH}/xvla-fix-comparison/8d

python -u run_inference.py \
    --task-filter "on_the_stove" \
    --max-episodes 1 \
    --predict-every 10 \
    --use-8d-state \
    --output-dir ${SCRATCH}/xvla-fix-comparison/8d \
    2>&1 | tee "${LOGDIR}/xvla-fix-8d.log"

echo ""
echo "=== Computing comparison metrics ==="
python -u << 'PYEOF'
import json, numpy as np, glob, os

scratch = os.environ["SCRATCH"]
DIM_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

def compute_stats(pred_actions, gt_actions):
    """Compute per-dimension stats."""
    results = {}
    for d in range(7):
        pv = np.array([p[d] for p in pred_actions])
        gv = np.array([g[d] for g in gt_actions])
        mae = np.mean(np.abs(pv - gv))
        std_ratio = np.std(pv) / max(np.std(gv), 1e-10)
        p = np.corrcoef(pv, gv)[0, 1] if np.std(pv) > 1e-10 and np.std(gv) > 1e-10 else float('nan')
        results[DIM_LABELS[d]] = {
            "mae": mae,
            "pred_std": np.std(pv),
            "gt_std": np.std(gv),
            "std_ratio": std_ratio,
            "pearson_r": p,
            "pred_mean": np.mean(pv),
            "gt_mean": np.mean(gv),
        }
    return results

for mode, label in [("20d", "20D state"), ("8d", "8D state (legacy)")]:
    json_files = glob.glob(f"{scratch}/xvla-fix-comparison/{mode}/*.json")
    if not json_files:
        print(f"No output files for {label}")
        continue

    all_preds = []
    all_gts = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        for ep in data["episodes"]:
            for pred in ep["predictions"]:
                all_preds.extend(pred["predicted_actions"])
                all_gts.extend(pred["ground_truth_actions"])

    print(f"\n{'='*60}")
    print(f"{label}: {len(all_preds)} prediction-action pairs")
    print(f"{'='*60}")
    print(f"{'Dim':>8s}  {'MAE':>8s}  {'PredStd':>8s}  {'GTStd':>8s}  {'StdRatio':>8s}  {'PearsonR':>8s}")
    print(f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    stats = compute_stats(all_preds, all_gts)
    for d, label_d in enumerate(DIM_LABELS):
        s = stats[label_d]
        print(f"{label_d:>8s}  {s['mae']:8.4f}  {s['pred_std']:8.4f}  {s['gt_std']:8.4f}  {s['std_ratio']:8.4f}  {s['pearson_r']:8.4f}")

    # Overall MAE for position dims (0-2)
    pos_mae = np.mean([stats[d]['mae'] for d in DIM_LABELS[:3]])
    rot_mae = np.mean([stats[d]['mae'] for d in DIM_LABELS[3:6]])
    print(f"\n  Position (x/y/z) avg MAE: {pos_mae:.4f}")
    print(f"  Rotation avg MAE: {rot_mae:.4f}")
    print(f"  Position avg StdRatio: {np.mean([stats[d]['std_ratio'] for d in DIM_LABELS[:3]]):.4f}")

print("\n=== Done ===")
PYEOF
SCRIPTEOF

echo "Submitting comparison test to long partition..."
sbatch --partition="long" "$BATCH_SCRIPT"
