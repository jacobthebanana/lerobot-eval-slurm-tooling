#!/usr/bin/env python3
"""
run_inference.py — Run X-VLA inference on the LIBERO-Spatial dataset (HDF5 format).

For each demonstration episode in the HDF5 files, predicts actions every
`--predict-every` timesteps. Each prediction produces a chunk of future actions
(chunk_size, typically 30). Saves raw data as JSON files (one per task) for later plotting.

Usage:
    python run_inference.py [--task-filter TASK] [--max-episodes N] [--predict-every N]

The dataset is loaded from HDF5 files under $SCRATCH/lerobot-demo-data/libero_spatial/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline

import numpy as np
import torch

# Note: We use direct 7D extraction (model was trained on zero-padded 7D LIBERO actions)

if TYPE_CHECKING:
    import h5py


# ---------------------------------------------------------------------------
# Rotation conversion utilities
# ---------------------------------------------------------------------------
def _quat_to_mat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion [..., x, y, z, w] to rotation matrix [..., 3, 3]."""
    x, y, z, w = (
        quat_xyzw[..., 0],
        quat_xyzw[..., 1],
        quat_xyzw[..., 2],
        quat_xyzw[..., 3],
    )
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.zeros(quat_xyzw.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[..., 0, 1] = 2.0 * (xy - wz)
    R[..., 0, 2] = 2.0 * (xz + wy)
    R[..., 1, 0] = 2.0 * (xy + wz)
    R[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[..., 1, 2] = 2.0 * (yz - wx)
    R[..., 2, 0] = 2.0 * (xz - wy)
    R[..., 2, 1] = 2.0 * (yz + wx)
    R[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def _axis_angle_to_mat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle [..., 3] to rotation matrix [..., 3, 3] using Rodriguez."""
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)  # (..., 1)
    mask = angle[..., 0] > 1e-10
    axis = np.zeros_like(axis_angle)
    axis[mask] = axis_angle[mask] / angle[mask]

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = np.cos(angle[..., 0])
    s = np.sin(angle[..., 0])
    C = 1.0 - c

    R = np.zeros(axis_angle.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = x * x * C + c
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = y * y * C + c
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = z * z * C + c
    return R


def _mat_to_rot6d(rot_mats: np.ndarray) -> np.ndarray:
    """Extract 6D rotation from batched rotation matrices [..., 3, 3] → [..., 6]."""
    col1 = rot_mats[..., :3, 0]  # first column
    col2 = rot_mats[..., :3, 1]  # second column
    return np.concatenate([col1, col2], axis=-1)


def build_state_20d(
    ee_pos: np.ndarray,
    robot_states: np.ndarray,
) -> np.ndarray:
    """
    Build the 20D state expected by X-VLA, matching the official LiberoProcessorStep.

    State layout (20D):
        [ee_x, ee_y, ee_z, rot6d(6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         └─ proprio (10) ──────────┘└─ zero-padding (10) ───────────────────────────┘

    The rotation is converted from quaternion (stored in robot_states) to a 6D
    continuous rotation representation (first two columns of the rotation matrix).

    Args:
        ee_pos: (T, 3) absolute end-effector positions.
        robot_states: (T, 9) array with format
            [gripper_q1, gripper_q2, ee_x, ee_y, ee_z, ee_qw, ee_qx, ee_qy, ee_qz].

    Returns:
        state_20d: (T, 20) float32 array.
    """
    T = ee_pos.shape[0]

    # Quaternion from robot_states columns 5:9 as [qw, qx, qy, qz]
    quat_wxyz = robot_states[:, 5:9].astype(np.float64)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]  # → [qx, qy, qz, qw]

    rot_mats = _quat_to_mat(quat_xyzw)
    rot6d = _mat_to_rot6d(rot_mats)

    return _assemble_state_20d(ee_pos, rot6d, T)


def build_state_20d_from_axis_angle(
    ee_pos: np.ndarray,
    ee_ori_axis_angle: np.ndarray,
) -> np.ndarray:
    """
    Build 20D state from axis-angle orientation (for LeRobot dataset compatibility).

    The LeRobot `lerobot/libero` dataset stores orientation as axis-angle in its
    8D state vector. This converts to 6D rotation and pads to 20D.

    Args:
        ee_pos: (T, 3) absolute end-effector positions.
        ee_ori_axis_angle: (T, 3) axis-angle orientation (in radians).

    Returns:
        state_20d: (T, 20) float32 array.
    """
    T = ee_pos.shape[0]
    rot_mats = _axis_angle_to_mat(ee_ori_axis_angle.astype(np.float64))
    rot6d = _mat_to_rot6d(rot_mats)
    return _assemble_state_20d(ee_pos, rot6d, T)


def _assemble_state_20d(
    ee_pos: np.ndarray,
    rot6d: np.ndarray,
    T: int,
) -> np.ndarray:
    """Assemble the final 20D state from position and 6D rotation components."""
    extra_zeros = np.zeros((T, 1), dtype=np.float32)
    proprio = np.concatenate(
        [ee_pos.astype(np.float32), rot6d.astype(np.float32), extra_zeros], axis=-1
    )
    state_20d = np.concatenate(
        [proprio, np.zeros_like(proprio, dtype=np.float32)], axis=-1
    )
    return state_20d.astype(np.float32)

# Force unbuffered output for SLURM logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRATCH = os.environ["SCRATCH"]
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(SCRATCH) / "xvla-inference-output" / "raw_data"
MODEL_ID = "lerobot/xvla-libero"

# HDF5 dataset location
HDF5_DIR = Path(SCRATCH) / "lerobot-demo-data" / "libero_spatial"


def sanitize_filename(name: str) -> str:
    """Convert a descriptive task name to a safe filename."""
    return name.strip().lower().replace(" ", "_").replace("/", "_").replace(".", "_")


def task_string_from_filename(filename: str) -> str:
    """Extract natural language task from HDF5 filename."""
    # e.g. "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5"
    # → "pick up the black bowl on the stove and place it on the plate"
    name = filename.replace("_demo.hdf5", "").replace(".hdf5", "")
    return name.replace("_", " ")


def convert_20d_to_7d(action_20d: np.ndarray) -> np.ndarray:
    """
    Extract LIBERO 7D delta action from X-VLA 20D output.

    The X-VLA model was trained with 7D LIBERO actions zero-padded to 20D:
        [dx, dy, dz, droll, dpitch, dyaw, gripper, 0, ..., 0]
    (indices 0:7 are the true action; indices 7:20 are zero-padding).

    The rotation at indices 3:6 is axis-angle directly (NOT 6D rotation),
    and the gripper at index 6 is a continuous value in [-1, 1] (trained via MSE).

    Returns 7D action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    was_1d = action_20d.ndim == 1
    if was_1d:
        action_20d = action_20d[np.newaxis, :]

    # Extract the first 7 dimensions directly
    action_7d = action_20d[..., :7].copy()

    # Threshold gripper (index 6) to binary -1/+1
    action_7d[..., 6] = np.where(action_7d[..., 6] > 0.0, 1.0, -1.0)

    if was_1d:
        action_7d = action_7d[0]
    return action_7d


def _build_observation(
    t: int,
    agentview: np.ndarray,
    eye_in_hand: np.ndarray,
    states: np.ndarray,
    task_name: str,
    flip_agentview: bool = True,
) -> dict[str, torch.Tensor | str]:
    """Build the observation dict expected by X-VLA's preprocessor for timestep `t`.

    Args:
        t: Timestep index.
        agentview: (T, H, W, 3) uint8 agentview images.
        eye_in_hand: (T, H, W, 3) uint8 eye-in-hand images.
        states: (T, state_dim) pre-built state vectors (20D with 6D rot, or 8D legacy).
        task_name: Language task description.
        flip_agentview: If True, flip agentview 180° (H and W) to match the
            LIBERO→VLA camera convention. The official LiberoProcessorStep does this
            because the LIBERO agentview camera is mounted upside-down relative to
            the HuggingFaceVLA convention used during training.
    """
    img1_np = agentview[t].astype(np.float32) / 255.0
    if flip_agentview:
        # Flip both H and W (180° rotation), matching LiberoProcessorStep
        img1_np = np.flip(img1_np, axis=(0, 1)).copy()
    img1 = torch.from_numpy(img1_np).permute(2, 0, 1)  # (C, H, W)

    img2_np = eye_in_hand[t].astype(np.float32) / 255.0
    img2 = torch.from_numpy(img2_np).permute(2, 0, 1)

    empty_cam = torch.zeros(3, 224, 224, dtype=torch.float32)
    state_t = torch.from_numpy(states[t].astype(np.float32))

    return {
        "observation.images.image": img1,
        "observation.images.image2": img2,
        "observation.images.empty_camera_0": empty_cam,
        "observation.state": state_t,
        "task": task_name,
    }


def _predict_at_timestep(
    t: int,
    agentview: np.ndarray,
    eye_in_hand: np.ndarray,
    states: np.ndarray,
    task_name: str,
    preprocess: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    policy: XVLAPolicy,
    postprocess: PolicyProcessorPipeline[Any, Any],
    chunk_size: int,
    gt_actions: np.ndarray,
    flip_agentview: bool = True,
) -> dict[str, Any] | None:
    """Run X-VLA prediction at a single timestep. Returns prediction dict or None."""
    try:
        raw_obs = _build_observation(
            t, agentview, eye_in_hand, states, task_name, flip_agentview=flip_agentview
        )
        batch = preprocess(raw_obs)

        with torch.inference_mode():
            pred_chunk = policy.predict_action_chunk(batch)

        pred_chunk = postprocess(pred_chunk)
        pred_chunk_np = pred_chunk.squeeze(0).cpu().numpy()  # (chunk_size, 20)
        pred_chunk_7d = convert_20d_to_7d(pred_chunk_np)  # (chunk_size, 7)

        gt_horizon = gt_actions[t : t + chunk_size]

        return {
            "start_t": int(t),
            "predicted_actions": pred_chunk_7d.tolist(),
            "ground_truth_actions": gt_horizon.tolist(),
        }
    except Exception as e:
        log.error(f"    Error at t={t}: {e}", exc_info=True)
        return None


def _process_demo(
    demo: h5py.Group,
    demo_key: str,
    task_name: str,
    predict_every: int,
    chunk_size: int,
    preprocess: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    policy: XVLAPolicy,
    postprocess: PolicyProcessorPipeline[Any, Any],
    use_20d_state: bool = True,
) -> dict[str, Any]:
    """Load data from one HDF5 demo group, run predictions, and return the result."""
    T = demo["actions"].shape[0]

    gt_actions = np.array(demo["actions"])  # (T, 7)

    # Build state vector — either 20D (6D rotation + padding, matching training)
    # or 8D (axis-angle + gripper, legacy behavior for comparison).
    if use_20d_state:
        # robot_states layout: [gripper_q1, gripper_q2, ee_x, ee_y, ee_z,
        #                       ee_qw, ee_qx, ee_qy, ee_qz]
        robot_states = np.array(demo["robot_states"])  # (T, 9)
        ee_pos = np.array(demo["obs/ee_pos"])  # (T, 3)
        states = build_state_20d(ee_pos, robot_states)  # (T, 20)
        state_dim = 20
    else:
        # Legacy 8D state for comparison
        ee_pos = np.array(demo["obs/ee_pos"])  # (T, 3)
        ee_ori = np.array(demo["obs/ee_ori"])  # (T, 3)
        gripper = np.array(demo["obs/gripper_states"])  # (T, 2)
        states = np.concatenate([ee_pos, ee_ori, gripper], axis=-1)  # (T, 8)
        state_dim = 8

    agentview = np.array(demo["obs/agentview_rgb"])  # (T, H, W, 3) uint8
    eye_in_hand = np.array(demo["obs/eye_in_hand_rgb"])

    # Flip agentview only when using 20D state (matching the official pipeline)
    flip_agentview = use_20d_state

    predictions = []
    for t in range(0, T, predict_every):
        pred = _predict_at_timestep(
            t,
            agentview,
            eye_in_hand,
            states,
            task_name,
            preprocess,
            policy,
            postprocess,
            chunk_size,
            gt_actions,
            flip_agentview=flip_agentview,
        )
        if pred is not None:
            predictions.append(pred)

    return {
        "demo_key": demo_key,
        "task_name": task_name,
        "episode_length": int(T),
        "ground_truth_actions": gt_actions.tolist(),
        "predictions": predictions,
        "chunk_size": chunk_size,
        "action_dim": 7,
        "state_dim": state_dim,
        "use_20d_state": use_20d_state,
    }


def _process_task_file(
    hdf5_path: Path,
    output_dir: str,
    model_id: str,
    predict_every: int,
    max_episodes_per_task: int | None,
    max_episodes: int | None,
    episodes_done: int,
    task_filter: set[str] | None,
    preprocess: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    policy: XVLAPolicy,
    postprocess: PolicyProcessorPipeline[Any, Any],
    chunk_size: int,
    use_20d_state: bool = True,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Process one HDF5 task file.  Returns (episode_results, new_episodes_done, hit_limit)."""
    import h5py

    task_name = task_string_from_filename(hdf5_path.name)
    safe_name = sanitize_filename(task_name)

    if task_filter and not any(tf in task_name.lower() for tf in task_filter):
        log.info(f"Skipping (filter): {task_name}")
        return [], episodes_done, False

    log.info(f"\n{'=' * 60}")
    log.info(f"Processing: {task_name}")
    log.info(f"File: {hdf5_path}")

    task_episodes: list[dict[str, Any]] = []

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[-1]),
        )

        for demo_key in demo_keys:
            if max_episodes_per_task and len(task_episodes) >= max_episodes_per_task:
                break
            if max_episodes and episodes_done >= max_episodes:
                break

            demo = f["data"][demo_key]
            log.info(f"  {demo_key}: {demo['actions'].shape[0]} timesteps")

            ep_result = _process_demo(
                demo,
                demo_key,
                task_name,
                predict_every,
                chunk_size,
                preprocess,
                policy,
                postprocess,
                use_20d_state=use_20d_state,
            )
            task_episodes.append(ep_result)
            episodes_done += 1

            log.info(f"    Done: {len(ep_result['predictions'])} predictions")

    # Save task results
    if task_episodes:
        output_path = os.path.join(output_dir, f"{safe_name}.json")
        output_data = {
            "task_name": task_name,
            "model_id": model_id,
            "chunk_size": chunk_size,
            "predict_every": predict_every,
            "num_episodes": len(task_episodes),
            "episodes": task_episodes,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        log.info(f"  Saved: {output_path} ({len(task_episodes)} episodes)")

    hit_limit = max_episodes is not None and episodes_done >= max_episodes
    return task_episodes, episodes_done, hit_limit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run X-VLA inference on LIBERO-Spatial HDF5"
    )
    parser.add_argument(
        "--task-filter",
        type=str,
        default=None,
        help="Comma-separated list of task name substrings to match (default: all).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit total episodes processed across all tasks (for testing).",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="Limit episodes per task.",
    )
    parser.add_argument(
        "--predict-every",
        type=int,
        default=10,
        help="Run prediction every N timesteps within each episode (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory for raw JSON output files.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(HDF5_DIR),
        help="Directory containing HDF5 task files.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=None,
        help="Override flow-matching denoising steps (default: use model config, 10).",
    )
    parser.add_argument(
        "--use-8d-state",
        action="store_true",
        default=False,
        help="Use legacy 8D state (axis-angle + gripper) instead of 20D state "
        "(6D rotation + padding). The 20D state matches the official lerobot-eval "
        "pipeline; 8D is the old run_inference.py behavior, kept for comparison.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    if device.type != "cuda":
        log.warning("CUDA not available — inference will be very slow on CPU!")

    # -----------------------------------------------------------------------
    # 1. Load X-VLA policy
    # -----------------------------------------------------------------------
    log.info(f"Loading X-VLA policy from '{args.model_id}' ...")

    policy = XVLAPolicy.from_pretrained(args.model_id).to(device).eval()
    chunk_size = policy.config.chunk_size
    log.info(
        f"Policy loaded. chunk_size={chunk_size}, "
        f"n_action_steps={policy.config.n_action_steps}, "
        f"max_action_dim={policy.config.max_action_dim}, "
        f"num_denoising_steps={policy.config.num_denoising_steps}"
    )

    # Optionally override denoising steps
    if args.denoising_steps is not None:
        log.info(f"Overriding denoising steps: {policy.config.num_denoising_steps} → {args.denoising_steps}")
        policy.config.num_denoising_steps = args.denoising_steps

    # Build pre/post processor pipeline from the serialized config
    log.info("Building pre/post processors ...")
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    log.info("Pre/post processors built.")

    # -----------------------------------------------------------------------
    # 2. Find HDF5 task files
    # -----------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        log.error(f"No .hdf5 files found in {data_dir}")
        sys.exit(1)
    log.info(f"Found {len(hdf5_files)} HDF5 files in {data_dir}")

    # Filter by task name
    task_filter = None
    if args.task_filter:
        # Accept both spaced and underscored filter terms
        raw_terms = [t.strip().lower() for t in args.task_filter.split(",")]
        task_filter = set()
        for t in raw_terms:
            task_filter.add(t)
            task_filter.add(t.replace("_", " "))  # also check with spaces

    # -----------------------------------------------------------------------
    # 3. Process each task file
    # -----------------------------------------------------------------------
    use_20d = not args.use_8d_state
    log.info(
        f"State format: {'20D (6D rotation + padding)' if use_20d else '8D (axis-angle + gripper, legacy)'}"
    )
    total_episodes_processed = 0

    for hdf5_path in hdf5_files:
        _, total_episodes_processed, hit_limit = _process_task_file(
            hdf5_path=hdf5_path,
            output_dir=args.output_dir,
            model_id=args.model_id,
            predict_every=args.predict_every,
            max_episodes_per_task=args.max_episodes_per_task,
            max_episodes=args.max_episodes,
            episodes_done=total_episodes_processed,
            task_filter=task_filter,
            preprocess=preprocess,
            policy=policy,
            postprocess=postprocess,
            chunk_size=chunk_size,
            use_20d_state=use_20d,
        )
        if hit_limit:
            log.info(f"Reached max_episodes={args.max_episodes}, stopping.")
            break

    log.info(f"\nDone! Processed {total_episodes_processed} episodes.")


if __name__ == "__main__":
    main()
