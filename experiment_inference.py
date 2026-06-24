#!/usr/bin/env python3
"""
experiment_inference.py — Test hypotheses for improving X-VLA ↔ LIBERO correlation.

Supports:
  --denoising-steps N   : Override flow-matching denoising steps (default: 10)
  --num-samples N       : Ensemble-average over N inference runs (default: 1)
  --sample-every N      : Predict every N timesteps (default: 10)
  --max-episodes N      : Limit episodes

Outputs a JSON metrics file with per-dimension correlation and error statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline

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

SCRATCH = os.environ["SCRATCH"]
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_ID = "lerobot/xvla-libero"
HDF5_DIR = Path(SCRATCH) / "lerobot-demo-data" / "libero_spatial"

DIM_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

# ---------------------------------------------------------------------------
# Rotation / state conversion (shared with run_inference.py)
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


def _mat_to_rot6d(rot_mats: np.ndarray) -> np.ndarray:
    """Extract 6D rotation from batched rotation matrices [..., 3, 3] → [..., 6]."""
    col1 = rot_mats[..., :3, 0]
    col2 = rot_mats[..., :3, 1]
    return np.concatenate([col1, col2], axis=-1)


def build_state_20d(ee_pos: np.ndarray, robot_states: np.ndarray) -> np.ndarray:
    """Build 20D state from quaternion in robot_states."""
    T = ee_pos.shape[0]
    quat_wxyz = robot_states[:, 5:9].astype(np.float64)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rot_mats = _quat_to_mat(quat_xyzw)
    rot6d = _mat_to_rot6d(rot_mats)
    extra = np.zeros((T, 1), dtype=np.float32)
    proprio = np.concatenate(
        [ee_pos.astype(np.float32), rot6d.astype(np.float32), extra], axis=-1
    )
    state_20d = np.concatenate(
        [proprio, np.zeros_like(proprio, dtype=np.float32)], axis=-1
    )
    return state_20d.astype(np.float32)


# ---------------------------------------------------------------------------
# Action conversion (FIXED: direct 7D extraction)
# ---------------------------------------------------------------------------
def convert_20d_to_7d(action_20d: np.ndarray) -> np.ndarray:
    """Extract first 7 dims from 20D X-VLA output (model trained on zero-padded 7D)."""
    was_1d = action_20d.ndim == 1
    if was_1d:
        action_20d = action_20d[np.newaxis, :]
    action_7d = action_20d[..., :7].copy()
    action_7d[..., 6] = np.where(action_7d[..., 6] > 0.0, 1.0, -1.0)
    if was_1d:
        action_7d = action_7d[0]
    return action_7d


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------
def _build_observation(
    t: int,
    agentview: np.ndarray,
    eye_in_hand: np.ndarray,
    states: np.ndarray,
    task_name: str,
    flip_agentview: bool = True,
) -> dict[str, torch.Tensor | str]:
    img1_np = agentview[t].astype(np.float32) / 255.0
    if flip_agentview:
        img1_np = np.flip(img1_np, axis=(0, 1)).copy()
    img1 = torch.from_numpy(img1_np).permute(2, 0, 1)
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


# ---------------------------------------------------------------------------
# Run a single prediction (with optional denoising step override)
# ---------------------------------------------------------------------------
def predict_single(
    t: int,
    agentview: np.ndarray,
    eye_in_hand: np.ndarray,
    states: np.ndarray,
    task_name: str,
    preprocess: PolicyProcessorPipeline,
    policy: XVLAPolicy,
    postprocess: PolicyProcessorPipeline,
    chunk_size: int,
    gt_actions: np.ndarray,
    denoising_steps: int | None = None,
    flip_agentview: bool = True,
) -> dict[str, Any] | None:
    """Run one prediction, optionally overriding denoising steps."""
    try:
        raw_obs = _build_observation(
            t, agentview, eye_in_hand, states, task_name, flip_agentview=flip_agentview
        )
        batch = preprocess(raw_obs)

        with torch.inference_mode():
            # Override denoising steps if requested
            if denoising_steps is not None:
                orig_steps = policy.config.num_denoising_steps
                policy.config.num_denoising_steps = denoising_steps

            pred_chunk = policy.predict_action_chunk(batch)

            if denoising_steps is not None:
                policy.config.num_denoising_steps = orig_steps

        pred_chunk = postprocess(pred_chunk)
        pred_chunk_np = pred_chunk.squeeze(0).cpu().numpy()
        pred_chunk_7d = convert_20d_to_7d(pred_chunk_np)

        gt_horizon = gt_actions[t : t + chunk_size]

        return {
            "start_t": int(t),
            "predicted_actions": pred_chunk_7d.tolist(),
            "ground_truth_actions": gt_horizon.tolist(),
        }
    except Exception as e:
        log.error(f"  Error at t={t}: {e}", exc_info=True)
        return None


def run_ensemble_prediction(
    t: int,
    agentview: np.ndarray,
    eye_in_hand: np.ndarray,
    states: np.ndarray,
    task_name: str,
    preprocess: PolicyProcessorPipeline,
    policy: XVLAPolicy,
    postprocess: PolicyProcessorPipeline,
    chunk_size: int,
    gt_actions: np.ndarray,
    denoising_steps: int | None = None,
    num_samples: int = 1,
    flip_agentview: bool = True,
) -> dict[str, Any] | None:
    """Average predictions over multiple inference runs for ensemble effect."""
    all_preds = []
    for _ in range(num_samples):
        pred = predict_single(
            t, agentview, eye_in_hand, states, task_name,
            preprocess, policy, postprocess, chunk_size, gt_actions,
            denoising_steps=denoising_steps,
            flip_agentview=flip_agentview,
        )
        if pred is not None:
            all_preds.append(np.array(pred["predicted_actions"]))

    if not all_preds:
        return None

    avg_pred = np.mean(all_preds, axis=0)
    gt_horizon = gt_actions[t : t + chunk_size]

    return {
        "start_t": int(t),
        "predicted_actions": avg_pred.tolist(),
        "ground_truth_actions": gt_horizon.tolist(),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(all_predictions: list[dict], gt_actions: np.ndarray) -> dict:
    """Compute per-dimension correlation and error metrics."""
    # Collect all (pred, gt) pairs
    pred_vals = {d: [] for d in range(7)}
    gt_vals = {d: [] for d in range(7)}

    for pred in all_predictions:
        pa = np.array(pred["predicted_actions"])
        ga = np.array(pred["ground_truth_actions"])
        min_len = min(len(pa), len(ga))
        for d in range(7):
            pred_vals[d].extend(pa[:min_len, d].tolist())
            gt_vals[d].extend(ga[:min_len, d].tolist())

    metrics = {}
    for d in range(7):
        pv = np.array(pred_vals[d])
        gv = np.array(gt_vals[d])

        mae = np.mean(np.abs(pv - gv))
        mse = np.mean((pv - gv) ** 2)

        # Pearson correlation
        p_std = np.std(pv)
        g_std = np.std(gv)
        if p_std > 1e-10 and g_std > 1e-10:
            pearson = np.corrcoef(pv, gv)[0, 1]
        else:
            pearson = float("nan")

        # Spearman rank correlation
        try:
            from scipy.stats import spearmanr
            spearman, _ = spearmanr(pv, gv)
        except ImportError:
            # Fallback: compute ranks manually
            def rankdata(x):
                temp = x.argsort()
                ranks = np.empty_like(temp, dtype=float)
                ranks[temp] = np.arange(len(x), dtype=float)
                return ranks
            rp = rankdata(pv)
            rg = rankdata(gv)
            rp_std = np.std(rp)
            rg_std = np.std(rg)
            if rp_std > 1e-10 and rg_std > 1e-10:
                spearman = np.corrcoef(rp, rg)[0, 1]
            else:
                spearman = float("nan")

        # Std ratio (pred/std vs gt/std)
        std_ratio = p_std / max(g_std, 1e-10)

        metrics[f"dim_{d}_{DIM_LABELS[d]}"] = {
            "mae": float(mae),
            "mse": float(mse),
            "pearson_r": float(pearson) if not np.isnan(pearson) else None,
            "spearman_r": float(spearman) if not np.isnan(spearman) else None,
            "pred_mean": float(np.mean(pv)),
            "pred_std": float(p_std),
            "gt_mean": float(np.mean(gv)),
            "gt_std": float(g_std),
            "std_ratio": float(std_ratio),
            "n_samples": len(pv),
        }

    # Overall metrics
    all_pv = np.concatenate([np.array(pred_vals[d]) for d in range(7)])
    all_gv = np.concatenate([np.array(gt_vals[d]) for d in range(7)])
    metrics["overall"] = {
        "mae": float(np.mean(np.abs(all_pv - all_gv))),
        "mse": float(np.mean((all_pv - all_gv) ** 2)),
    }

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="X-VLA inference experiments")
    parser.add_argument("--denoising-steps", type=int, default=10,
                        help="Flow-matching denoising steps (default: 10)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of ensemble samples to average (default: 1)")
    parser.add_argument("--sample-every", type=int, default=10,
                        help="Predict every N timesteps (default: 10)")
    parser.add_argument("--max-episodes", type=int, default=5,
                        help="Max total episodes (default: 5)")
    parser.add_argument("--task-filter", type=str, default="from_table_center",
                        help="Task substring filter (default: from_table_center)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(SCRATCH) / "xvla-experiments"),
                        help="Output directory for metrics JSON")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use-8d-state", action="store_true", default=False,
                        help="Use legacy 8D state instead of 20D (matching official pipeline)")
    args = parser.parse_args()

    use_20d = not args.use_8d_state

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Config: denoising_steps={args.denoising_steps}, "
             f"num_samples={args.num_samples}, sample_every={args.sample_every}")
    log.info(f"State format: {'20D (6D rotation)' if use_20d else '8D (axis-angle, legacy)'}")

    # Load policy
    log.info(f"Loading {MODEL_ID} ...")
    policy = XVLAPolicy.from_pretrained(MODEL_ID).to(device).eval()
    chunk_size = policy.config.chunk_size
    log.info(f"Loaded. chunk_size={chunk_size}, "
             f"n_action_steps={policy.config.n_action_steps}")

    preprocess, postprocess = make_pre_post_processors(
        policy.config, MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # Find HDF5 files
    import h5py
    hdf5_files = sorted(HDF5_DIR.glob("*.hdf5"))
    if not hdf5_files:
        log.error(f"No HDF5 files in {HDF5_DIR}")
        sys.exit(1)

    # Filter by task
    task_filter = args.task_filter.lower().replace("_", " ")

    all_predictions = []
    all_gt_actions = []
    episode_count = 0
    flip_agentview = use_20d

    for hdf5_path in hdf5_files:
        task_name = hdf5_path.stem.replace("_demo", "").replace("_", " ").replace(".hdf5", "")
        if task_filter not in task_name.lower():
            continue

        log.info(f"Processing: {task_name}")
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(
                [k for k in f["data"].keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[-1]),
            )

            for demo_key in demo_keys:
                if args.max_episodes and episode_count >= args.max_episodes:
                    break

                demo = f["data"][demo_key]
                T = demo["actions"].shape[0]
                log.info(f"  {demo_key}: {T} steps")

                gt_actions = np.array(demo["actions"])
                ee_pos = np.array(demo["obs/ee_pos"])

                if use_20d:
                    robot_states = np.array(demo["robot_states"])
                    states = build_state_20d(ee_pos, robot_states)
                else:
                    ee_ori = np.array(demo["obs/ee_ori"])
                    gripper_states = np.array(demo["obs/gripper_states"])
                    states = np.concatenate([ee_pos, ee_ori, gripper_states], axis=-1)

                agentview = np.array(demo["obs/agentview_rgb"])
                eye_in_hand = np.array(demo["obs/eye_in_hand_rgb"])

                episode_preds = []
                for t in range(0, T, args.sample_every):
                    if args.num_samples > 1:
                        pred = run_ensemble_prediction(
                            t, agentview, eye_in_hand, states, task_name,
                            preprocess, policy, postprocess, chunk_size, gt_actions,
                            denoising_steps=args.denoising_steps,
                            num_samples=args.num_samples,
                            flip_agentview=flip_agentview,
                        )
                    else:
                        pred = predict_single(
                            t, agentview, eye_in_hand, states, task_name,
                            preprocess, policy, postprocess, chunk_size, gt_actions,
                            denoising_steps=args.denoising_steps,
                            flip_agentview=flip_agentview,
                        )
                    if pred is not None:
                        episode_preds.append(pred)

                all_predictions.extend(episode_preds)
                all_gt_actions.append(gt_actions)
                episode_count += 1
                log.info(f"    {len(episode_preds)} predictions")

            if args.max_episodes and episode_count >= args.max_episodes:
                break

        if args.max_episodes and episode_count >= args.max_episodes:
            break

    # Compute metrics
    log.info(f"Computing metrics over {len(all_predictions)} predictions "
             f"from {episode_count} episodes...")

    combined_gt = np.concatenate(all_gt_actions, axis=0)
    metrics = compute_metrics(all_predictions, combined_gt)

    # Add config info
    metrics["config"] = {
        "denoising_steps": args.denoising_steps,
        "num_samples": args.num_samples,
        "sample_every": args.sample_every,
        "num_episodes": episode_count,
        "num_predictions": len(all_predictions),
        "model_id": MODEL_ID,
        "task_filter": args.task_filter,
        "seed": args.seed,
    }

    # Save
    exp_name = f"ds{args.denoising_steps}_ns{args.num_samples}_se{args.sample_every}"
    output_path = os.path.join(args.output_dir, f"metrics_{exp_name}.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics to {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS: denoising_steps={args.denoising_steps}, "
          f"num_samples={args.num_samples}")
    print(f"{'='*70}")
    for d in range(7):
        key = f"dim_{d}_{DIM_LABELS[d]}"
        m = metrics[key]
        print(f"  {DIM_LABELS[d]:8s}: MAE={m['mae']:.4f}  "
              f"Pearson={m['pearson_r']:.4f}  "
              f"Spearman={m['spearman_r']:.4f}  "
              f"StdRatio={m['std_ratio']:.4f}")
    print(f"  {'OVERALL':8s}: MAE={metrics['overall']['mae']:.4f}  "
          f"MSE={metrics['overall']['mse']:.4f}")


if __name__ == "__main__":
    main()
