#!/usr/bin/env bash
# submit_all_jobs.sh — Submit all evaluation jobs with NFS-safe venv extraction
#
# Extracts venv tarball to /tmp/$SLURM_JOB_ID/venv/ at runtime to avoid NFS
# stale-file race conditions. Uses the venv's python binary directly rather
# than relying on 'source activate' for PATH resolution.
#
set -euo pipefail

SCRATCH="${SCRATCH}"
PROJECT_DIR="${HOME}/lerobot-eval-slurm-toolings"

PARTITION="long"
TIMEOUT="24:00:00"
N_EPISODES=50

# Log directories
PI05_LOG_DIR="${SCRATCH}/pi05-eval-logs-v4"
PI0FAST_LOG_DIR="${SCRATCH}/pi0fast-eval-logs"
PI0_LOG_DIR="${SCRATCH}/pi0-eval-logs"
PI05_FT_LOG_DIR="${SCRATCH}/pi05-finetuned-eval-logs"
XVLA_LOG_DIR="${SCRATCH}/xvla-google-eval-logs-v3"
VLAJEPA_LOG_DIR="${SCRATCH}/vlajepa-eval-logs-v4"
FASTWAM_LOG_DIR="${SCRATCH}/fastwam-eval-logs"
VQBET_LOG_DIR="${SCRATCH}/vqbet-eval-logs-v4"
SMOLVLA_LOG_DIR="${SCRATCH}/smolvla-eval-logs"

mkdir -p "$PI05_LOG_DIR" "$PI0FAST_LOG_DIR" "$PI0_LOG_DIR" "$PI05_FT_LOG_DIR" \
         "$XVLA_LOG_DIR" "$VLAJEPA_LOG_DIR" "$FASTWAM_LOG_DIR" "$VQBET_LOG_DIR" \
         "$SMOLVLA_LOG_DIR"

echo "=========================================="
echo "  Submitting all evaluation jobs"
echo "  Date: $(date)"
echo "=========================================="

ALL_JOBS=()

# ==================================================================
# generate_and_submit
# ==================================================================
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

    # Build constraint line
    local constraint_line=""
    if [ -n "${constraints}" ]; then
        constraint_line="#SBATCH --constraint=${constraints}"
    fi

    # Write batch script. Use explicit python path instead of source activate.
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

# Extract venv to local tmp
echo "=== Extracting venv to local tmp ==="
tar -xzf "${venv_tarball}" -C "\$LOCAL_TMP"
VENV="\$LOCAL_TMP/${venv_name}"
PYTHON="\${VENV}/bin/python"

# Validate extraction
if [ ! -f "\$PYTHON" ]; then
    echo "FATAL: Python binary not found at \$PYTHON"
    echo "  Tarball: ${venv_tarball}"
    echo "  Extraction dir: \$LOCAL_TMP"
    ls -la "\$VENV/bin/" 2>/dev/null || echo "  No bin/ directory found"
    exit 1
fi
echo "  Venv extracted to: \${VENV}"
echo "  Python: \${PYTHON}"

# Extra environment setup (vulkan)
${extra_env_setup}

# LIBERO config (safe even for non-LIBERO tasks)
LIBERO_CONFIG_DIR="\${HOME}/.libero"
LIBERO_CONFIG_FILE="\${LIBERO_CONFIG_DIR}/config.yaml"
LIBERO_PKG_DIR="\${VENV}/lib/${python_version}/site-packages/libero/libero"
LIBERO_DATASETS_DIR="${SCRATCH}/libero-datasets"

if [ ! -f "\${LIBERO_CONFIG_FILE}" ]; then
    echo "--- Creating LIBERO config ---"
    mkdir -p "\${LIBERO_CONFIG_DIR}" "\${LIBERO_DATASETS_DIR}"
    cat > "\${LIBERO_CONFIG_FILE}" << LIBERO_EOF
benchmark_root: \${LIBERO_PKG_DIR}
bddl_files: \${LIBERO_PKG_DIR}/bddl_files
init_states: \${LIBERO_PKG_DIR}/init_files
datasets: \${LIBERO_DATASETS_DIR}
assets: \${LIBERO_PKG_DIR}/assets
LIBERO_EOF
    echo "  Created \${LIBERO_CONFIG_FILE}"
fi

# Use explicit python path + env activation
source "\${VENV}/bin/activate"

echo "=== Environment ==="
echo "Host: \$(hostname)"
echo "Python: \${PYTHON}"
echo "Python version: \$(\${PYTHON} --version)"
echo "CUDA available: \$(\${PYTHON} -c 'import torch; print(torch.cuda.is_available())')"

# Detect MuJoCo rendering backend
echo ""
echo "--- Detecting MuJoCo rendering backend ---"
MUJOCO_BACKEND=""
if MUJOCO_GL=egl \$PYTHON -c "import robosuite" 2>/dev/null; then
    MUJOCO_BACKEND="egl"
    echo "  Using EGL backend (GPU-accelerated)"
elif MUJOCO_GL=osmesa \$PYTHON -c "import robosuite" 2>/dev/null; then
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

# Use explicit python binary
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

# ==================================================================
# Group A
# ==================================================================
echo ""
echo "=== Group A: LIBERO policies ==="

LIBERO_TARBALL="${SCRATCH}/uv-venv/lerobot-libero-eval-v2.tar.gz"
LIBERO_VENV="lerobot-libero-eval-v2"
LIBERO_PY="python3.10"

# pi05-libero (5 subsets, batch_size=1, use_amp=true)
PI05_OUTPUT_BASE="${SCRATCH}/pi05-eval-output-v4/$(date +%Y%m%d_%H%M%S)"
echo ""
echo "-- pi05-libero --"
for suite in libero_spatial libero_object libero_goal libero_10 libero_90; do
    case "${suite}" in
        libero_spatial) short="pi05-spatial" ;;
        libero_object)  short="pi05-object" ;;
        libero_goal)    short="pi05-goal" ;;
        libero_10)      short="pi05-10" ;;
        libero_90)      short="pi05-90" ;;
    esac
    generate_and_submit \
        "${short}" \
        "${PI05_LOG_DIR}" \
        "${LIBERO_TARBALL}" \
        "${LIBERO_VENV}" \
        "${LIBERO_PY}" \
        "" \
        "\${PYTHON} -m lerobot.scripts.lerobot_eval \
            --policy.path=lerobot/pi05-libero \
            --env.type=libero \
            --env.task=${suite} \
            --env.obs_type=pixels_agent_pos \
            --env.control_mode=relative \
            --eval.n_episodes=${N_EPISODES} \
            --eval.batch_size=1 \
            --policy.device=cuda \
            --policy.use_amp=true \
            --seed=1000 \
            --output_dir=${PI05_OUTPUT_BASE}/${suite}" \
        "${PI05_OUTPUT_BASE}/${suite}" \
        "48gb"
done

# pi0fast-libero (spatial only)
PI0FAST_OUTPUT_BASE="${SCRATCH}/pi0fast-eval-output/$(date +%Y%m%d_%H%M%S)"
echo ""
echo "-- pi0fast-libero --"
generate_and_submit \
    "pi0fast-spatial" \
    "${PI0FAST_LOG_DIR}" \
    "${LIBERO_TARBALL}" \
    "${LIBERO_VENV}" \
    "${LIBERO_PY}" \
    "" \
    "\${PYTHON} -m lerobot.scripts.lerobot_eval \
        --policy.path=lerobot/pi0fast-libero \
        --env.type=libero \
        --env.task=libero_spatial \
        --env.obs_type=pixels_agent_pos \
        --env.control_mode=relative \
        --eval.n_episodes=${N_EPISODES} \
        --eval.batch_size=10 \
        --policy.device=cuda \
        --policy.use_amp=false \
        --seed=1000 \
        --output_dir=${PI0FAST_OUTPUT_BASE}/libero_spatial" \
    "${PI0FAST_OUTPUT_BASE}/libero_spatial" \
    "48gb"

# pi0_libero_finetuned_v044
PI0_OUTPUT_BASE="${SCRATCH}/pi0-eval-output/$(date +%Y%m%d_%H%M%S)"
echo ""
echo "-- pi0_libero_finetuned_v044 --"
generate_and_submit \
    "pi0-finetuned-spatial" \
    "${PI0_LOG_DIR}" \
    "${LIBERO_TARBALL}" \
    "${LIBERO_VENV}" \
    "${LIBERO_PY}" \
    "" \
    "\${PYTHON} -m lerobot.scripts.lerobot_eval \
        --policy.path=lerobot/pi0_libero_finetuned_v044 \
        --env.type=libero \
        --env.task=libero_spatial \
        --env.obs_type=pixels_agent_pos \
        --env.control_mode=relative \
        --eval.n_episodes=${N_EPISODES} \
        --eval.batch_size=10 \
        --policy.device=cuda \
        --policy.use_amp=false \
        --seed=1000 \
        --output_dir=${PI0_OUTPUT_BASE}/libero_spatial" \
    "${PI0_OUTPUT_BASE}/libero_spatial" \
    "48gb"

# pi05_libero_finetuned_v044
PI05_FT_OUTPUT_BASE="${SCRATCH}/pi05-finetuned-eval-output/$(date +%Y%m%d_%H%M%S)"
echo ""
echo "-- pi05_libero_finetuned_v044 --"
generate_and_submit \
    "pi05-finetuned-spatial" \
    "${PI05_FT_LOG_DIR}" \
    "${LIBERO_TARBALL}" \
    "${LIBERO_VENV}" \
    "${LIBERO_PY}" \
    "" \
    "\${PYTHON} -m lerobot.scripts.lerobot_eval \
        --policy.path=lerobot/pi05_libero_finetuned_v044 \
        --env.type=libero \
        --env.task=libero_spatial \
        --env.obs_type=pixels_agent_pos \
        --env.control_mode=relative \
        --eval.n_episodes=${N_EPISODES} \
        --eval.batch_size=10 \
        --policy.device=cuda \
        --policy.use_amp=false \
        --seed=1000 \
        --output_dir=${PI05_FT_OUTPUT_BASE}/libero_spatial" \
    "${PI05_FT_OUTPUT_BASE}/libero_spatial" \
    "48gb"

# ==================================================================
# Group B
# ==================================================================
echo ""
echo "=== Group B: SimplerEnv / XVLA ==="

SIMPLERENV_TARBALL="${SCRATCH}/uv-venv/simplerenv.tar.gz"
SIMPLERENV_VENV="simplerenv"
SIMPLERENV_PY="python3.10"
SIMPLERENV_EXTRA='export LD_LIBRARY_PATH="${SCRATCH}/local-libs/vulkan/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"'

XVLA_SCRIPT="${PROJECT_DIR}/eval_xvla_google_robot.py"
XVLA_OUTPUT_BASE="${SCRATCH}/xvla-google-eval-output-v3/$(date +%Y%m%d_%H%M%S)"

echo ""
echo "-- xvla-google-robot --"
for task in google_robot_pick_coke_can google_robot_move_near google_robot_open_drawer google_robot_close_drawer; do
    short="${task#google_robot_}"
    generate_and_submit \
        "xvla-${short}" \
        "${XVLA_LOG_DIR}" \
        "${SIMPLERENV_TARBALL}" \
        "${SIMPLERENV_VENV}" \
        "${SIMPLERENV_PY}" \
        "${SIMPLERENV_EXTRA}" \
        "\${PYTHON} ${XVLA_SCRIPT} \
            --task=${task} \
            --n-episodes=${N_EPISODES} \
            --max-steps=200 \
            --device=cuda \
            --seed=42 \
            --output-dir=${XVLA_OUTPUT_BASE}/${short}" \
        "${XVLA_OUTPUT_BASE}/${short}"
done

# ==================================================================
# Group C
# ==================================================================
echo ""
echo "=== Group C: Newer LeRobot policies ==="

NEW_TARBALL="${SCRATCH}/uv-venv/lerobot-new.tar.gz"
NEW_VENV="lerobot-new"
NEW_PY="python3.12"

# CUDA 13.0 driver NVRTC fix: ensure nvidia/cu13/lib is on LD_LIBRARY_PATH
# PyTorch 2.11.0 bundles CUDA 12.8 libs, but CUDA 13.0 driver needs cu13 NVRTC
CUDA13_NVRTC_FIX='export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"'

# VLA-JEPA-LIBERO (5 subsets)
VLAJEPA_OUTPUT_BASE="${SCRATCH}/vlajepa-eval-output-v4/$(date +%Y%m%d_%H%M%S)"
echo ""
echo "-- VLA-JEPA-LIBERO --"
for suite in libero_spatial libero_object libero_goal libero_10 libero_90; do
    case "${suite}" in
        libero_spatial) short="vlajepa-spatial" ;;
        libero_object)  short="vlajepa-object" ;;
        libero_goal)    short="vlajepa-goal" ;;
        libero_10)      short="vlajepa-10" ;;
        libero_90)      short="vlajepa-90" ;;
    esac
    generate_and_submit \
        "${short}" \
        "${VLAJEPA_LOG_DIR}" \
        "${NEW_TARBALL}" \
        "${NEW_VENV}" \
        "${NEW_PY}" \
        "${CUDA13_NVRTC_FIX}" \
        "\${PYTHON} -m lerobot.scripts.lerobot_eval \
            --policy.path=lerobot/VLA-JEPA-LIBERO \
            --env.type=libero \
            --env.task=${suite} \
            --env.obs_type=pixels_agent_pos \
            --env.control_mode=relative \
            --eval.n_episodes=${N_EPISODES} \
            --eval.batch_size=10 \
            --eval.use_async_envs=false \
            --policy.device=cuda \
            --policy.use_amp=false \
            --seed=1000 \
            --output_dir=${VLAJEPA_OUTPUT_BASE}/${suite}" \
        "${VLAJEPA_OUTPUT_BASE}/${suite}" \
        "ampere"
done

# SmolVLA-LIBERO (5 subsets, 0.5B compact VLA, consumer-grade)
SMOLVLA_OUTPUT_BASE="${SCRATCH}/smolvla-eval-output/$(date +%Y%m%d_%H%M%S)"
echo ""
echo "-- SmolVLA-LIBERO --"
for suite in libero_spatial libero_object libero_goal libero_10 libero_90; do
    case "${suite}" in
        libero_spatial) short="smolvla-spatial" ;;
        libero_object)  short="smolvla-object" ;;
        libero_goal)    short="smolvla-goal" ;;
        libero_10)      short="smolvla-10" ;;
        libero_90)      short="smolvla-90" ;;
    esac
    generate_and_submit \
        "${short}" \
        "${SMOLVLA_LOG_DIR}" \
        "${NEW_TARBALL}" \
        "${NEW_VENV}" \
        "${NEW_PY}" \
        "${CUDA13_NVRTC_FIX}" \
        "\${PYTHON} -m lerobot.scripts.lerobot_eval \
            --policy.path=lerobot/smolvla_libero \
            --env.type=libero \
            --env.task=${suite} \
            --env.obs_type=pixels_agent_pos \
            --env.control_mode=relative \
            --eval.n_episodes=${N_EPISODES} \
            --eval.batch_size=10 \
            --policy.device=cuda \
            --policy.use_amp=true \
            --seed=1000 \
            --output_dir=${SMOLVLA_OUTPUT_BASE}/${suite}" \
        "${SMOLVLA_OUTPUT_BASE}/${suite}" \
        "48gb"
done

# FastWAM
FASTWAM_OUTPUT_BASE="${SCRATCH}/fastwam-eval-output/$(date +%Y%m%d_%H%M%S)"
echo ""
echo "-- FastWAM --"
generate_and_submit \
    "fastwam-spatial" \
    "${FASTWAM_LOG_DIR}" \
    "${NEW_TARBALL}" \
    "${NEW_VENV}" \
    "${NEW_PY}" \
    "${CUDA13_NVRTC_FIX}" \
    "\${PYTHON} -m lerobot.scripts.lerobot_eval \
        --policy.path=lerobot/fastwam_libero_uncond_2cam224 \
        --env.type=libero \
        --env.task=libero_spatial \
        --env.obs_type=pixels_agent_pos \
        --env.control_mode=relative \
        --eval.n_episodes=${N_EPISODES} \
        --eval.batch_size=10 \
        --policy.device=cuda \
        --policy.use_amp=false \
        --seed=1000 \
        --output_dir=${FASTWAM_OUTPUT_BASE}/libero_spatial" \
    "${FASTWAM_OUTPUT_BASE}/libero_spatial" \
    "80gb"

# VQ-BeT Pusht
VQBET_OUTPUT_BASE="${SCRATCH}/vqbet-eval-output-v4/$(date +%Y%m%d_%H%M%S)"
VQBET_LOCAL="${SCRATCH}/local-models/vqbet_pusht_fixed"
echo ""
echo "-- VQ-BeT Pusht --"
generate_and_submit \
    "vqbet-pusht" \
    "${VQBET_LOG_DIR}" \
    "${NEW_TARBALL}" \
    "${NEW_VENV}" \
    "${NEW_PY}" \
    "${CUDA13_NVRTC_FIX}" \
    "\${PYTHON} -m lerobot.scripts.lerobot_eval \
        --policy.path=${VQBET_LOCAL} \
        --env.type=pusht \
        --env.task=PushT-v0 \
        --eval.n_episodes=50 \
        --eval.batch_size=5 \
        --policy.device=cuda \
        --policy.use_amp=false \
        --seed=1000 \
        --output_dir=${VQBET_OUTPUT_BASE}" \
    "${VQBET_OUTPUT_BASE}"

# ==================================================================
# Summary
# ==================================================================
echo ""
echo "=========================================="
echo "  Summary: ${#ALL_JOBS[@]} jobs submitted"
echo "=========================================="
for j in "${ALL_JOBS[@]}"; do
    echo "  ${j}"
done
echo ""
echo "Track: squeue -u \$USER"
