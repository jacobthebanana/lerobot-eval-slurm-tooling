# LeRobot Evaluation Pipeline on Mila Cluster — Setup Report

_Generated with Claude Code CLI (deepseek-v4-pro[1m])_

## 1. Overview

This report documents the end-to-end process of setting up SLURM-based evaluation runs for HuggingFace LeRobot policies on the Mila cluster. The pipeline evaluates vision-language-action policies (X-VLA, PI05, PI0, FastWAM, VQ-BeT) on simulation environments (LIBERO, PushT, SimplerEnv / Google Robot).

---

## 2. Infrastructure

### 2.1 Virtual Environments

Three purpose-specific virtual environments were created under `$SCRATCH/uv-venv/`:

| Venv                     | Python | Key Packages                                                 | Purpose                               |
| ------------------------ | ------ | ------------------------------------------------------------ | ------------------------------------- |
| `lerobot-libero-eval-v2` | 3.10   | lerobot 0.4.4, mujoco 3.1.6, robosuite 1.4.0                 | pi05, pi0, pi0fast, pi05-ft on LIBERO |
| `lerobot-new`            | 3.12   | lerobot 0.5.2 (feat/add-fastwam), mujoco 3.1.6, torch 2.11.0 | VLA-JEPA, FastWAM, VQ-BeT on LIBERO   |
| `simplerenv`             | 3.10   | lerobot[xvla], SAPIEN 2.2.2, SimplerEnv                      | xvla-google-robot on SimplerEnv       |

**Why separate venvs:** Different policies require different LeRobot versions (v0.4.4 vs v0.5.2) and dependency combinations. Installing `gym-aloha` downgrades mujoco in ways that break LIBERO. Isolation prevents cross-contamination.

### 2.2 Tarball Packaging

Each venv is packaged as a gzipped tarball on `$SCRATCH`:

```
lerobot-libero-eval-v2.tar.gz   (4.7 GB)
lerobot-new.tar.gz              (4.8 GB)
simplerenv.tar.gz               (5.5 GB)
```

### 2.3 Storage Layout

```
$SCRATCH
├── uv-venv/                    # Virtual environments
├── hf-cache/                   # HuggingFace model cache
├── libero-datasets/            # LIBERO benchmark assets
├── local-models/               # Locally patched model configs
│   └── vqbet_pusht_fixed/      # VQ-BeT with fixed config
├── *-eval-logs-v5/             # Job stdout/stderr (per policy)
├── *-eval-output-v5/           # Evaluation results (JSON, videos)
└── local-libs/vulkan/          # Vulkan libraries for SimplerEnv
```

---

## 3. SLURM Submission Infrastructure

### 3.1 `submit_all_jobs.sh`

The main submission script at `submit_all_jobs.sh` generates per-job SLURM batch scripts via the `generate_and_submit()` function.

**Key parameters:**

```bash
PARTITION="long"          # 7-day max, no resource limits
TIMEOUT="24:00:00"        # 24-hour wall time
N_EPISODES=50             # Evaluation episodes per task
```

**Function signature:**

```bash
generate_and_submit() {
    local job_name="$1"          # SLURM job name
    local log_dir="$2"           # stdout/stderr directory
    local venv_tarball="$3"      # path to .tar.gz
    local venv_name="$4"         # directory name inside tarball
    local python_version="$5"    # e.g. "python3.10", "python3.12"
    local extra_env_setup="$6"   # e.g. LD_LIBRARY_PATH for vulkan/nvrtc
    local eval_command="$7"      # the python command to run
    local output_dir="$8"        # results directory
    local constraints="${9:-}"   # SLURM --constraint (optional)
}
```

### 3.2 NFS Race Condition Fix

The Mila cluster's NFS storage exhibits stale-file issues: files installed via `pip` on the login node may not be visible on compute nodes for minutes to hours. This caused `FileNotFoundError` on random package files (torch, triton, httpx, sympy, etc.).

**Solution:** Package the venv as a tarball on shared storage, extract to local `/tmp/$SLURM_JOB_ID/` on each compute node at job start:

```bash
LOCAL_TMP="/tmp/${SLURM_JOB_ID}"
mkdir -p "$LOCAL_TMP"
tar -xzf "${SCRATCH}/uv-venv/lerobot-new.tar.gz" -C "$LOCAL_TMP"
VENV="$LOCAL_TMP/lerobot-new"
PYTHON="${VENV}/bin/python"
```

The `source activate` command was patched to prevent it from redirecting `VIRTUAL_ENV` back to the NFS path:

```bash
source "${VENV}/bin/activate"
```

### 3.3 MuJoCo Rendering Backend Detection

LIBERO requires either EGL (GPU-accelerated) or OSMesa (software) rendering. The script auto-detects:

```bash
if MUJOCO_GL=egl ${PYTHON} -c "import robosuite" 2>/dev/null; then
    MUJOCO_BACKEND="egl"
elif MUJOCO_GL=osmesa ${PYTHON} -c "import robosuite" 2>/dev/null; then
    MUJOCO_BACKEND="osmesa"
else
    MUJOCO_BACKEND="egl"  # fallback
fi
export MUJOCO_GL="${MUJOCO_BACKEND}"
export PYOPENGL_PLATFORM="${MUJOCO_BACKEND}"
```

### 3.4 LIBERO Configuration

A `~/.libero/config.yaml` is generated at job start pointing to paths inside the extracted venv:

```yaml
benchmark_root: ${VENV}/lib/python3.12/site-packages/libero/libero
bddl_files: ${VENV}/lib/python3.12/site-packages/libero/libero/bddl_files
init_states: ${VENV}/lib/python3.12/site-packages/libero/libero/init_files
datasets: ${SCRATCH}/libero-datasets
assets: ${VENV}/lib/python3.12/site-packages/libero/libero/assets
```

---

## 4. GPU Hardware Research

_redacted_

---

## 5. Issues, Root Causes, Attempted Fixes, and Solutions

### 5.1 NFS Stale-File Race Condition

**Symptom:** `FileNotFoundError` on random package files across all venvs. Failed packages included torch, triton, mpmath, sympy, httpx, datasets.

**Root cause:** `pip install` writes to NFS from the login node, but compute nodes' NFS caches may serve stale directory entries for minutes to hours.

**Solution:** Package venvs as gzipped tarballs, extract to local `/tmp/$SLURM_JOB_ID/` on each compute node. This guarantees the compute node sees a consistent filesystem state.

**Code:**

```bash
tar -xzf "${SCRATCH}/uv-venv/lerobot-new.tar.gz" -C "$LOCAL_TMP"
VENV="$LOCAL_TMP/lerobot-new"
source "${VENV}/bin/activate"
```

---

### 5.2 VLA-JEPA CUDA Kernel Architecture Mismatch

**Symptom:**

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

**Root cause:** The V-JEPA2 ViT-L encoder (`facebook/vjepa2-vitl-fpc64-256`) ships custom CUDA kernels compiled only for compute capability ≥ 8.0 (A100+). The cluster has V100 (sm_70), RTX 8000 (sm_75), and A100 (sm_80) nodes. Without GPU constraints, SLURM may allocate V100 or RTX 8000 nodes whose GPUs cannot execute these kernels.

**Attempted fixes (failed):**

1. Using NFS venv directly — same error, different GPU architecture
2. Using local venv extraction — same error
3. Trying different PyTorch versions — not a PyTorch issue, specific to V-JEPA's custom kernels

**Solution:** Add `--constraint=ampere` to SLURM job submission, restricting to A100 (sm_80) and A6000 (sm_86) GPUs:

```bash
#SBATCH --constraint=ampere
```

**Verification:** After applying the constraint, the VLA-JEPA model loads successfully on A100 nodes. The `cudaErrorNoKernelImageForDevice` error no longer occurs.

---

### 5.3 VLA-JEPA NVRTC Library Missing (CUDA 13.0 Driver)

**Symptom:**

```
nvrtc: error: failed to open libnvrtc-builtins.so.13.0.
  Make sure that libnvrtc-builtins.so.13.0 is installed correctly.
```

**Root cause:** The Mila cluster runs NVIDIA driver 580.159.03 with CUDA 13.0. PyTorch 2.11.0 ships with CUDA 12.8 runtime libraries (in `nvidia/cuda_nvrtc/lib/`). When the V-JEPA encoder performs JIT compilation of CUDA kernels (e.g., `reduction_prod_kernel`), the CUDA 13.0 driver invokes NVRTC which looks for `libnvrtc-builtins.so.13.0`. PyTorch bundles `nvidia/cu13` libraries alongside `nvidia/cuda_nvrtc`, but `nvidia/cu13/lib/` is not on the dynamic linker's search path by default.

The CUDA 13.0 NVRTC libraries exist at:

```
${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc-builtins.so.13.0
${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc.so.13
```

But the linker only resolves them if `LD_LIBRARY_PATH` includes this directory.

**Solution:** Add `nvidia/cu13/lib` to `LD_LIBRARY_PATH` before starting evaluation:

```bash
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
```

**Verification:** After applying the fix, VLA-JEPA evaluation runs for 30+ minutes producing evaluation videos without the NVRTC error. Zero `nvrtc` matches in stderr logs.

---

### 5.4 PI05/PI0/PI0Fast Transformers Version Mismatch

**Symptom:**

```
ValueError: An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues
```

**Root cause:** LeRobot v0.4.4 (in `lerobot-libero-eval-v2` venv) expects a specific transformers version that includes a custom `transformers.models.siglip.check` module. This module is injected by LeRobot's "transformers replace" mechanism during installation. With transformers 4.57.6 installed, the `check` module is absent, triggering the error in `modeling_pi05.py:579`:

```python
from transformers.models.siglip import check
if not check.check_whether_transformers_replace_is_installed_correctly():
    msg = """An incorrect transformer version is used..."""
    raise ValueError(msg)
```

**Solution:** Use the `lerobot-new` venv (LeRobot v0.5.2 from `feat/add-fastwam` branch, Python 3.12). This newer version removes the `check` module dependency and uses a different PI05 implementation that loads without the transformers version check.

**Verification:** Import test confirms:

```python
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # OK
from lerobot.policies.pi05.modeling_pi05 import PI05Policy        # OK
# No transformers error
```

**Trade-off:** This venv uses Python 3.12 (vs 3.10 in `lerobot-libero-eval-v2`) and requires the NVRTC fix described in §5.3.

---

### 5.5 PI05 Native bf16 Support

**Symptom:** PI05 model uses `torch_dtype: bfloat16` in its config. Turing GPUs (RTX 8000, sm_75) lack native bf16 support and emulate it in software, causing slow inference.

**Solution:** Use `--constraint=ampere` for PI05 jobs as well, ensuring A100/A6000 GPUs with native bf16 hardware support:

```bash
#SBATCH --constraint=ampere
```

---

### 5.6 PI05/PI0 OOM on 32GB GPUs

**Symptom:** `torch.cuda.OutOfMemoryError` on V100 32GB when running pi05 with `batch_size=10`.

**Root cause:** The PI05 model (OpenPI-based diffusion policy) requires significant VRAM for the denoising process and environment rendering.

**Attempted fixes:**

1. `--eval.batch_size=1` — reduced memory but still failed with transformers error

**Solution:** Use `--constraint=48gb` (or `ampere`/`80gb`) to target GPUs with ≥48GB VRAM. Combined with `--eval.batch_size=1` for the largest suites (libero_90).

---

### 5.7 SimplerEnv SAPIEN SIGSEGV

**Symptom:**

```
Segmentation fault (core dumped)
```

and in earlier logs:

```
GLFW error: X11: The DISPLAY environment variable is missing
```

**Root cause:** SAPIEN 2.2.2 (used by SimplerEnv for Google Robot simulation) requires a Vulkan-capable display server. Without an X11 display, GLFW initialization crashes.

**Attempted fixes (failed):**

1. Downloaded `libvulkan.so.1` to `$SCRATCH/local-libs/vulkan/` and set `LD_LIBRARY_PATH` — still crashed
2. Tried various combinations of `PYOPENGL_PLATFORM`, `EGL_PLATFORM`, and `DISPLAY` variables

**Status:** **UNRESOLVED.** SimplerEnv evaluation requires either Xvfb (virtual framebuffer) or SAPIEN offscreen rendering configuration. The Google Robot evaluation jobs produce a secondary error (action shape mismatch `((1,7), 7)` in `convert_20d_to_7d()`) but crash with SIGSEGV before reaching that code path.

---

### 5.8 FastWAM OOM

**Symptom:** FastWAM (5B gemma + 1B action head model) fails to fit on A100 80GB with `batch_size=10`.

**Status:** **UNRESOLVED.** Even with `--constraint=80gb` targeting A100 80GB, the model exceeds available VRAM. Requires model sharding, quantization, or larger GPU (H100 80GB, which is restricted to `short-unkillable` partition with 3-hour limit).

---

### 5.9 VLA-JEPA Async Environments Deadlock

**Symptom:** Model loads but produces zero output for 45+ minutes — no actions, no errors.

**Root cause:** LIBERO's async environment vectorization (`use_async_envs=True`) deadlocks with VLA-JEPA's multi-step inference pattern.

**Solution:** Disable async environments:

```bash
--eval.use_async_envs=false
```

---

### 5.10 VQ-BeT Config Mismatch

**Symptom:**

```
The fields 'mlp_hidden_dim' are not valid for VQBeTConfig
```

**Root cause:** The HuggingFace checkpoint `lerobot/vqbet_pusht` uses a config format that includes `mlp_hidden_dim`, which is not recognized by the installed LeRobot version's `VQBeTConfig`.

**Solution:** Download the model config, strip the `mlp_hidden_dim` field, and create a local model directory:

```
${SCRATCH}/local-models/vqbet_pusht_fixed/
```

**Status:** Downloads succeed but evaluation fails with `ConnectionResetError` (HF Hub download issue on compute nodes).

---

### 5.11 MuJoCo 3.10.0 Compatibility

**Symptom:**

```
TypeError: mj_fullM() missing required argument
```

**Root cause:** MuJoCo 3.10.0 removed the legacy `mj_fullM` function, but robosuite 1.4.0 requires it.

**Solution:** Pin mujoco to 3.1.x:

```bash
uv pip install "mujoco>=3.1,<3.2"
```

---

### 5.12 Silent Evaluation (No Log Output Under SLURM)

**Symptom:** Jobs run for 30+ minutes with no stdout/stderr output after "Assets already downloaded."

**Root cause:** The LeRobot evaluation framework (`lerobot/scripts/lerobot_eval.py`) deliberately disables tqdm progress bars when `SLURM_JOB_ID` is set:

```python
# lerobot/utils/utils.py:38-41
def inside_slurm():
    return "SLURM_JOB_ID" in os.environ

# lerobot/scripts/lerobot_eval.py:341
trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
```

The `rollout()` and `eval_policy()` functions contain zero `print()`, `logging.info()`, or `logging.debug()` calls. Under SLURM, the only real-time evidence of progress is:

- Video files being written to disk
- GPU utilization via `nvidia-smi`

**Status:** This is expected behavior, not a bug. The silence is by design. Results appear at job completion via `print()` statements for aggregated metrics and `eval_info.json`.

---

## 6. Current Evaluation Status (2026-06-24)

_redacted_

---

## 7. Proven Evaluation Command Templates

### VLA-JEPA on LIBERO (Ampere GPU required)

```bash
python -m lerobot.scripts.lerobot_eval \
    --policy.path=lerobot/VLA-JEPA-LIBERO \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.obs_type=pixels_agent_pos \
    --env.control_mode=relative \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --eval.use_async_envs=false \
    --policy.device=cuda \
    --policy.use_amp=false \
    --seed=1000 \
    --output_dir=/path/to/output
```

**Critical requirements:**

- `--constraint=ampere` (A100/A6000, sm_80+)
- `LD_LIBRARY_PATH` must include `nvidia/cu13/lib`
- `--eval.use_async_envs=false` (prevents deadlock)
- `--policy.use_amp=false` (VLA-JEPA uses native bf16)

### PI05 on LIBERO (48GB+ GPU, prefer Ampere for bf16)

```bash
python -m lerobot.scripts.lerobot_eval \
    --policy.path=lerobot/pi05-libero \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.obs_type=pixels_agent_pos \
    --env.control_mode=relative \
    --eval.n_episodes=50 \
    --eval.batch_size=1 \
    --policy.device=cuda \
    --policy.use_amp=true \
    --seed=1000 \
    --output_dir=/path/to/output
```

**Critical requirements:**

- `--constraint=ampere` (for native bf16 support)
- Use `lerobot-new` venv (LeRobot v0.5.2, no transformers check)
- `--eval.batch_size=1` (to avoid OOM on libero_90)

---

## 8. Key Files

| File                        | Purpose                                                      |
| --------------------------- | ------------------------------------------------------------ |
| `submit_all_jobs.sh`        | Main submission script with `generate_and_submit()`          |
| `slurm_batch_template.sh`   | Standalone template (not used by submitter; reference)       |
| `eval_xvla_google_robot.py` | Custom SimplerEnv evaluation script                          |
| `test_xvla_google_robot.py` | Model loading/preprocessing verification                     |
| `install_simplerenv.sh`     | SimplerEnv + SAPIEN installer                                |
| `CLAUDE.md`                 | Project instructions (store large files on $SCRATCH, use uv) |
