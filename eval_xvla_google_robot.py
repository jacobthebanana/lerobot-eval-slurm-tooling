#!/usr/bin/env python3
"""
eval_xvla_google_robot.py -- Closed-loop SimplerEnv evaluation for lerobot/xvla-google-robot.

Evaluates the Google Robot checkpoint of X-VLA on SimplerEnv simulation tasks.

Requirements (installed via install_simplerenv.sh):
    - SimplerEnv (https://github.com/simpler-env/SimplerEnv)
    - lerobot[xvla]
    - torch (with CUDA)

Usage:
    # Single task
    python eval_xvla_google_robot.py --task google_robot_pick_coke_can --n-episodes 20

    # Multiple tasks
    python eval_xvla_google_robot.py \
        --task google_robot_pick_coke_can,google_robot_move_near \
        --n-episodes 50

    # With video recording
    python eval_xvla_google_robot.py --task google_robot_pick_coke_can --record-videos

    # With GPU
    python eval_xvla_google_robot.py --task google_robot_pick_coke_can --device cuda

Available SimplerEnv Google Robot tasks:
    - google_robot_pick_coke_can
    - google_robot_pick_object
    - google_robot_move_near
    - google_robot_open_drawer
    - google_robot_close_drawer
    - google_robot_place_in_closed_drawer

    Subtask variants (task--subtask):
    - google_robot_pick_coke_can--fls
    - google_robot_pick_coke_can--vertical
    - google_robot_pick_coke_can--standing
    - google_robot_open_drawer--top
    - google_robot_open_drawer--middle
    - google_robot_open_drawer--bottom
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "lerobot/xvla-google-robot"
SCRATCH = os.environ["SCRATCH"]

# SimplerEnv Google Robot tasks
SIMPLERENV_GOOGLE_ROBOT_TASKS = [
    "google_robot_pick_coke_can",
    "google_robot_pick_object",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_place_in_closed_drawer",
]

# Task variants (with subtask qualifier)
SIMPLERENV_SUBTASK_VARIANTS = [
    "google_robot_pick_coke_can--fis",
    "google_robot_pick_coke_can--vertical",
    "google_robot_pick_coke_can--standing",
    "google_robot_open_drawer--top",
    "google_robot_open_drawer--middle",
    "google_robot_open_drawer--bottom",
    "google_robot_close_drawer--top",
    "google_robot_close_drawer--middle",
    "google_robot_close_drawer--bottom",
]

# ---------------------------------------------------------------------------
# Rotation conversion utilities
# ---------------------------------------------------------------------------


def rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation (two columns of rotation matrix) to axis-angle.

    Args:
        rot6d: (..., 6) array where rot6d[..., :3] is first column, rot6d[..., 3:] is second column.

    Returns:
        (..., 3) axis-angle representation.
    """
    try:
        from lerobot.policies.xvla.utils import rotate6d_to_axis_angle

        return rotate6d_to_axis_angle(rot6d)
    except ImportError:
        pass

    # Fallback: reconstruct rotation matrix, convert to axis-angle
    col1 = rot6d[..., :3]
    col2 = rot6d[..., 3:6]
    col3 = np.cross(col1, col2, axis=-1)

    R = np.stack([col1, col2, col3], axis=-1)  # (..., 3, 3)

    # Rotation matrix to axis-angle
    # Compute angle from trace
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))

    # Compute axis from antisymmetric part
    axis_x = R[..., 2, 1] - R[..., 1, 2]
    axis_y = R[..., 0, 2] - R[..., 2, 0]
    axis_z = R[..., 1, 0] - R[..., 0, 1]
    axis = np.stack([axis_x, axis_y, axis_z], axis=-1)

    # Normalize axis
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    mask = axis_norm[..., 0] > 1e-10
    axis_masked = np.where(mask[..., None], axis / np.clip(axis_norm, 1e-10, None), 0.0)

    return axis_masked * angle[..., None]


def convert_20d_to_7d(action_20d: np.ndarray) -> np.ndarray:
    """Convert XVLA 20D ee6d action to SimplerEnv 7D action.

    XVLA ee6d action layout (20D):
        [0:3]   - delta xyz (position)
        [3:9]   - delta rotation (6D continuous representation)
        [9]     - gripper
        [10:20] - padding / alternative (not used for single-arm)

    SimplerEnv action layout (7D):
        [0:3]   - delta xyz
        [3:6]   - delta rotation (axis-angle)
        [6]     - gripper (-1 open, +1 close)

    Args:
        action_20d: (..., 20) model output.

    Returns:
        (..., 7) SimplerEnv action.
    """
    # Ensure we have a batch dimension for uniform processing
    was_1d = action_20d.ndim == 1
    if was_1d:
        action_20d = action_20d[np.newaxis, :]

    # Extract components from first group (indices 0-9)
    delta_xyz = action_20d[..., 0:3]  # (..., 3)
    rot_6d = action_20d[..., 3:9]  # (..., 6)

    # Convert 6D rotation to axis-angle
    rot_aa = rot6d_to_axis_angle(rot_6d)  # (..., 3)

    # Extract gripper (index 9), threshold to binary
    gripper = action_20d[..., 9:10]  # (..., 1)
    gripper = np.where(gripper > 0.0, 1.0, -1.0)

    # Assemble 7D action
    action_7d = np.concatenate([delta_xyz, rot_aa, gripper], axis=-1)

    # Remove batch dimension if present (SimplerEnv expects unbatched actions)
    if was_1d or action_7d.shape[0] == 1:
        action_7d = action_7d[0]
    return action_7d


def _extract_image_from_simpler_obs(env, sim_obs: dict) -> np.ndarray:
    """Extract camera image from SimplerEnv/ManiSkill2 observation.

    Uses the SimplerEnv utility function if available, otherwise falls back
    to raw observation dict extraction.
    """
    try:
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

        return get_image_from_maniskill2_obs_dict(env, sim_obs)
    except (ImportError, Exception):
        pass

    # Fallback: try common observation keys
    for key in ["image", "Image", "rgb", "RGB", "camera", "Camera"]:
        if key in sim_obs:
            img = sim_obs[key]
            if isinstance(img, np.ndarray):
                return img

    log.warning("No image found in SimplerEnv observation; using zeros.")
    return np.zeros((256, 256, 3), dtype=np.uint8)


def _extract_state_from_simpler_obs(sim_obs: dict) -> np.ndarray:
    """Extract robot state from SimplerEnv/ManiSkill2 observation.

    ManiSkill2 Google Robot state format (from 'agent' dict):
        - qpos: joint positions (8D: 7 arm joints + 1 gripper)

    Returns 8D state vector matching what the XVLA google-robot expects.
    """
    agent = sim_obs.get("agent", None)
    if agent is None:
        log.warning("No 'agent' key in observation; searching for state...")
        for key in ["qpos", "state", "robot_state", "proprio"]:
            if key in sim_obs:
                val = np.asarray(sim_obs[key], dtype=np.float32)
                if val.ndim == 0:
                    val = val.reshape(1)
                if val.ndim > 1:
                    val = val.flatten()
                # Pad or truncate to 8D
                if len(val) < 8:
                    val = np.pad(val, (0, 8 - len(val)), mode="constant")
                elif len(val) > 8:
                    val = val[:8]
                return val
        log.warning("No state found; using zeros.")
        return np.zeros(8, dtype=np.float32)

    # ManiSkill2 'agent' is a dict with 'qpos' key
    if isinstance(agent, dict):
        if "qpos" in agent:
            state_np = np.asarray(agent["qpos"], dtype=np.float32)
        else:
            # Try to concatenate all numeric values
            values = []
            for v in agent.values():
                if isinstance(v, (int, float)):
                    values.append(float(v))
                elif isinstance(v, np.ndarray) and v.ndim == 1:
                    values.extend(v.tolist())
            state_np = np.array(values, dtype=np.float32) if values else np.zeros(8, dtype=np.float32)
    else:
        state_np = np.asarray(agent, dtype=np.float32)

    if state_np.ndim > 1:
        state_np = state_np.flatten()

    # Pad or truncate to 8D
    if len(state_np) < 8:
        state_np = np.pad(state_np, (0, 8 - len(state_np)), mode="constant")
    elif len(state_np) > 8:
        state_np = state_np[:8]

    return state_np


def _normalize_image(img: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Normalize image to [0,1] float32 and resize to target size."""
    import cv2

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype != np.float32:
        img = img.astype(np.float32)

    # Handle grayscale
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=-1)

    # Ensure HWC format
    if img.shape[0] == 3 and img.shape[1] != 3 and img.ndim == 3:
        # Assume CHW → HWC
        img = img.transpose(1, 2, 0)

    # Resize if needed
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    return img


def build_observation(
    env,
    sim_obs: dict,
    language_instruction: str,
) -> dict[str, torch.Tensor | str]:
    """Convert SimplerEnv observation to XVLA policy observation format.

    SimplerEnv observations (ManiSkill2 format):
        - agent: dict with 'qpos' (8D joint positions)
        - image: camera image (H, W, 3) uint8

    XVLA google-robot expects:
        - observation.images.image: (3, 256, 256) float32 [0,1]
        - observation.images.image2: (3, 256, 256) float32 [0,1]
        - observation.images.empty_camera_0: (3, 224, 224) float32 zeros
        - observation.state: (8,) float32
        - task: str language instruction

    Args:
        env: SimplerEnv instance.
        sim_obs: SimplerEnv observation dict.
        language_instruction: Task description string.

    Returns:
        Dict ready for policy preprocessing.
    """
    # Extract and normalize images
    img_np = _extract_image_from_simpler_obs(env, sim_obs)
    img_np = _normalize_image(img_np, target_size=(256, 256))

    # SimplerEnv provides only 1 camera; duplicate for image2
    img1 = torch.from_numpy(img_np).permute(2, 0, 1).float()  # (3, 256, 256)
    img2 = img1.clone()
    empty_cam = torch.zeros(3, 224, 224, dtype=torch.float32)

    # Extract state
    state_np = _extract_state_from_simpler_obs(sim_obs)
    state_t = torch.from_numpy(state_np).float()

    return {
        "observation.images.image": img1,
        "observation.images.image2": img2,
        "observation.images.empty_camera_0": empty_cam,
        "observation.state": state_t,
        "task": language_instruction,
    }


# ---------------------------------------------------------------------------
# Eval functions
# ---------------------------------------------------------------------------


def run_episode(
    env,
    policy,
    preprocessor,
    max_steps: int = 200,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run a single episode in SimplerEnv with the XVLA policy.

    Args:
        env: SimplerEnv environment.
        policy: XVLAPolicy instance.
        preprocessor: Policy preprocessor pipeline.
        max_steps: Maximum number of steps per episode.
        device: Torch device.

    Returns:
        Dict with episode results.
    """
    policy.reset()

    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()

    episode_reward = 0.0
    done = False
    truncated = False
    step = 0
    actions_taken = []
    success = False

    while not (done or truncated) and step < max_steps:
        # Build observation in model format
        model_obs = build_observation(env, obs, instruction)

        # Preprocess
        batch = preprocessor(model_obs)

        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # Policy step
        with torch.inference_mode():
            action_20d = policy.select_action(batch)

        # Convert 20D ee6d → 7D SimplerEnv action
        action_np = action_20d.cpu().numpy()
        action_7d = convert_20d_to_7d(action_np)

        # Apply action to environment
        obs, reward, done, truncated, info = env.step(action_7d)
        episode_reward += reward
        actions_taken.append(action_7d.tolist())
        step += 1

        # Check for success
        if info.get("is_success", False) or info.get("success", False):
            success = True

    return {
        "success": success,
        "reward": episode_reward,
        "steps": step,
        "n_actions": len(actions_taken),
        "instruction": instruction,
    }


def evaluate(
    task_name: str,
    n_episodes: int = 50,
    max_steps: int = 200,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> dict:
    """Evaluate the XVLA google-robot policy on a SimplerEnv task.

    Args:
        task_name: SimplerEnv task name (e.g., 'google_robot_pick_coke_can').
        n_episodes: Number of episodes to run.
        max_steps: Maximum steps per episode.
        device: Torch device.
        seed: Random seed.

    Returns:
        Dict with evaluation metrics.
    """
    import simpler_env

    log.info(f"Creating SimplerEnv for task: {task_name}")
    env = simpler_env.make(task_name)

    log.info(f"Loading XVLA google-robot model: {MODEL_ID}")
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    policy = XVLAPolicy.from_pretrained(MODEL_ID).to(device).eval()
    log.info(f"Model loaded. chunk_size={policy.config.chunk_size}")

    # Build preprocessor from the model's saved config
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    log.info(f"Preprocessor steps: {[s.__class__.__name__ for s in preprocessor.steps]}")

    # Run episodes
    np.random.seed(seed)
    torch.manual_seed(seed)

    episodes = []
    successes = []
    rewards = []
    steps_per_episode = []
    start_time = time.time()

    for ep in range(n_episodes):
        ep_result = run_episode(env, policy, preprocessor, max_steps=max_steps, device=device)
        episodes.append(ep_result)
        successes.append(ep_result["success"])
        rewards.append(ep_result["reward"])
        steps_per_episode.append(ep_result["steps"])

        if (ep + 1) % 5 == 0 or ep == n_episodes - 1:
            current_rate = np.mean(successes) * 100
            log.info(
                f"  Episode {ep + 1}/{n_episodes}: "
                f"success={ep_result['success']}, "
                f"reward={ep_result['reward']:.2f}, "
                f"steps={ep_result['steps']}, "
                f"running_success_rate={current_rate:.1f}%"
            )

    elapsed = time.time() - start_time
    success_rate = np.mean(successes) * 100

    env.close()

    results = {
        "model_id": MODEL_ID,
        "task": task_name,
        "n_episodes": n_episodes,
        "success_rate": float(success_rate),
        "success_count": int(sum(successes)),
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps_per_episode)),
        "std_steps": float(np.std(steps_per_episode)),
        "total_time_s": round(elapsed, 2),
        "time_per_episode_s": round(elapsed / n_episodes, 3),
        "seed": seed,
        "per_episode": [
            {
                "episode": i,
                "success": bool(s),
                "reward": float(r),
                "steps": int(st),
            }
            for i, (s, r, st) in enumerate(zip(successes, rewards, steps_per_episode))
        ],
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate X-VLA google-robot on SimplerEnv simulation tasks"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="google_robot_pick_coke_can",
        help="SimplerEnv task name (comma-separated for multiple tasks)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of episodes per task",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(SCRATCH) / "xvla-eval-output"/ "google_robot"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    log.info(f"Using device: {device}")
    log.info(f"Output directory: {args.output_dir}")

    # Parse tasks
    tasks = [t.strip() for t in args.task.split(",")]

    all_results = {}
    for task in tasks:
        log.info(f"\n{'='*70}")
        log.info(f"Evaluating task: {task}")
        log.info(f"{'='*70}")

        try:
            results = evaluate(
                task_name=task,
                n_episodes=args.n_episodes,
                max_steps=args.max_steps,
                device=device,
                seed=args.seed,
            )
            all_results[task] = results

            # Print summary
            log.info(f"\n--- Results for {task} ---")
            log.info(f"  Success rate: {results['success_rate']:.1f}%")
            log.info(f"  Avg reward:   {results['avg_reward']:.2f}")
            log.info(f"  Avg steps:    {results['avg_steps']:.1f}")
            log.info(f"  Time/episode: {results['time_per_episode_s']:.2f}s")

        except Exception as e:
            log.error(f"Failed to evaluate {task}: {e}", exc_info=True)
            all_results[task] = {"error": str(e)}

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"eval_results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults saved to: {output_path}")

    # Print overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    for task, results in all_results.items():
        if "error" in results:
            print(f"  {task}: ERROR - {results['error']}")
        else:
            print(f"  {task}: {results['success_rate']:.1f}% success "
                  f"({results['success_count']}/{results['n_episodes']})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
