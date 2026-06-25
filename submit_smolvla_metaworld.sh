#!/usr/bin/env bash
# submit_smolvla_metaworld.sh — Submit SmolVLA-MetaWorld evaluation jobs
set -euo pipefail

SCRATCH="${SCRATCH}"
PROJECT_DIR="${HOME}/lerobot-eval-slurm-toolings"

PARTITION="long"
TIMEOUT="24:00:00"
N_EPISODES=50

SMOLVLA_MW_LOG_DIR="${SCRATCH}/smolvla-metaworld-eval-logs"
mkdir -p "$SMOLVLA_MW_LOG_DIR"

NEW_TARBALL="${SCRATCH}/uv-venv/lerobot-new.tar.gz"
NEW_VENV="lerobot-new"
NEW_PY="python3.12"

CUDA13_NVRTC_FIX='export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"'

ALL_JOBS=()

generate_and_submit() {
    local job_name="$1"
    local log_dir="$2"
    local venv_tarball="$3"
    local venv_name="$4"
    local python_version="$5"
    local extra_env_setup="$6"
    local eval_command="$7"
    local output_dir="$8"
    local constraints="${9:-}"

    local script_file
    script_file=$(mktemp /tmp/slurm_${job_name}.XXXX.sh)

    local constraint_line=""
    if [ -n "${constraints}" ]; then
        constraint_line="#SBATCH --constraint=${constraints}"
    fi

    cat > "$script_file" << SCRIPTEOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_dir}/${job_name}-%j.out
#SBATCH --error=${log_dir}/${job_name}-%j.err
#SBATCH --time=${TIMEOUT}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=${PARTITION}
${constraint_line}

set -euo pipefail

LOCAL_TMP="/tmp/\${SLURM_JOB_ID}"
mkdir -p "\$LOCAL_TMP"

export HF_HOME="${SCRATCH}/hf-cache"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${SCRATCH}/xdg-cache"

echo "=== Extracting venv to local tmp ==="
tar -xzf "${venv_tarball}" -C "\$LOCAL_TMP"
VENV="\$LOCAL_TMP/${venv_name}"
PYTHON="\${VENV}/bin/python"

if [ ! -f "\$PYTHON" ]; then
    echo "FATAL: Python binary not found at \$PYTHON"
    ls -la "\$VENV/bin/" 2>/dev/null || echo "  No bin/ directory found"
    exit 1
fi
echo "  Venv extracted to: \${VENV}"
echo "  Python: \${PYTHON}"

${extra_env_setup}

source "\${VENV}/bin/activate"

echo "=== Environment ==="
echo "Host: \$(hostname)"
echo "Python: \${PYTHON}"
echo "Python version: \$(\${PYTHON} --version)"
echo "CUDA available: \$(\${PYTHON} -c 'import torch; print(torch.cuda.is_available())')"

echo ""
echo "--- Detecting MuJoCo rendering backend ---"
MUJOCO_BACKEND=""
if MUJOCO_GL=egl \$PYTHON -c "import metaworld" 2>/dev/null; then
    MUJOCO_BACKEND="egl"
    echo "  Using EGL backend (GPU-accelerated)"
elif MUJOCO_GL=osmesa \$PYTHON -c "import metaworld" 2>/dev/null; then
    MUJOCO_BACKEND="osmesa"
    echo "  Using OSMesa backend (software rendering)"
else
    MUJOCO_BACKEND="egl"
    echo "  WARNING: No rendering backend detected, trying EGL"
fi

export MUJOCO_GL="\${MUJOCO_BACKEND}"
export PYOPENGL_PLATFORM="\${MUJOCO_BACKEND}"

if [ "\${MUJOCO_BACKEND}" = "egl" ]; then
    for egl_dir in /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d /usr/lib/x86_64-linux-gnu/GL; do
        if [ -d "\$egl_dir" ]; then
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

${eval_command}

echo ""
echo "=== Eval complete ==="
echo "Results in: ${output_dir}"

rm -rf "\$LOCAL_TMP"
echo "  Cleaned up \${LOCAL_TMP}"
SCRIPTEOF

    chmod +x "$script_file"
    echo "--- Submitting ${job_name} ---"
    local job_out
    job_out=$(sbatch --partition="${PARTITION}" "$script_file" 2>&1)
    echo "  ${job_out}"
    local job_id
    job_id=$(echo "${job_out}" | grep -oP '\d+')
    ALL_JOBS+=("${job_id} (${job_name})")
    rm -f "$script_file"
}

echo "=========================================="
echo "  Submitting SmolVLA-MetaWorld evaluation"
echo "  Date: $(date)"
echo "=========================================="

SMOLVLA_MW_OUTPUT_BASE="${SCRATCH}/smolvla-metaworld-eval-output/$(date +%Y%m%d_%H%M%S)"
echo "  Output base: ${SMOLVLA_MW_OUTPUT_BASE}"

# MetaWorld difficulty suites: easy (28 tasks), medium (11 tasks),
# hard (6 tasks), very_hard (5 tasks).
# Each suite runs as a separate SLURM job.
for suite in easy medium hard very_hard; do
    case "${suite}" in
        easy)      short="smolvla-mw-easy" ;;
        medium)    short="smolvla-mw-medium" ;;
        hard)      short="smolvla-mw-hard" ;;
        very_hard) short="smolvla-mw-veryhard" ;;
    esac
    generate_and_submit \
        "${short}" \
        "${SMOLVLA_MW_LOG_DIR}" \
        "${NEW_TARBALL}" \
        "${NEW_VENV}" \
        "${NEW_PY}" \
        "${CUDA13_NVRTC_FIX}" \
        "\${PYTHON} -m lerobot.scripts.lerobot_eval \
            --policy.path=lerobot/smolvla_metaworld \
            --env.type=metaworld \
            --env.task=${suite} \
            --env.obs_type=pixels_agent_pos \
            --rename_map='{\"observation.image\": \"observation.images.camera1\"}' \
            --policy.empty_cameras=2 \
            --eval.n_episodes=${N_EPISODES} \
            --eval.batch_size=10 \
            --policy.device=cuda \
            --policy.use_amp=true \
            --seed=1000 \
            --output_dir=${SMOLVLA_MW_OUTPUT_BASE}/${suite}" \
        "${SMOLVLA_MW_OUTPUT_BASE}/${suite}" \
        ""
done

echo ""
echo "=========================================="
echo "  Summary: ${#ALL_JOBS[@]} SmolVLA-MetaWorld jobs submitted"
echo "  Output base: ${SMOLVLA_MW_OUTPUT_BASE}"
echo "=========================================="
for j in "${ALL_JOBS[@]}"; do
    echo "  ${j}"
done
echo ""
echo "Track: squeue -u \$USER | grep smolvla-mw"
echo "Logs:  ls ${SMOLVLA_MW_LOG_DIR}/"
echo "Output: ls ${SMOLVLA_MW_OUTPUT_BASE}/"
