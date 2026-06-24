#!/usr/bin/env python3
"""
plot_results.py — Visualize X-VLA predictions vs. LIBERO-Spatial ground truth.

Reads raw JSON files from output/raw_data/ and produces per-dimension plots
in output/dim_0/ through output/dim_6/.

Each plot shows:
  - Ground truth action values as a solid blue line over the full episode.
  - Predicted action chunks (30 steps each) as orange/red trajectories,
    each registered as a separate series starting from a prediction point.
  - Filled circles marking where each prediction starts (blue on ground truth,
    orange/red on the first point of each predicted trajectory).

Usage:
    python plot_results.py [--input-dir raw_data/] [--output-dir output/] [--dim D]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
GROUND_TRUTH_COLOR = "#2166ac"       # blue
PREDICTION_COLOR = "#d73027"         # red
PREDICTION_ALPHA = 0.35
MARKER_SIZE = 36
PRED_MARKER = "o"
GT_MARKER = "o"

# Labels for each LIBERO action dimension
DIM_LABELS = [
    "dx (delta x)",
    "dy (delta y)",
    "dz (delta z)",
    "droll (delta roll)",
    "dpitch (delta pitch)",
    "dyaw (delta yaw)",
    "gripper",
]


def make_plot(
    episode_data: dict,
    dim: int,
    output_path: str,
    show_prediction_start: bool = True,
):
    """
    Create a single-dimension plot for one episode or task.

    Parameters
    ----------
    episode_data : dict
        Contains 'ground_truth_actions' and 'predictions'.
    dim : int
        Which action dimension to plot (0-6).
    output_path : str
        Where to save the PNG.
    show_prediction_start : bool
        Whether to draw filled-circle markers at prediction start points.
    """
    gt_actions = np.array(episode_data["ground_truth_actions"])  # (T, A)
    predictions = episode_data["predictions"]
    chunk_size = episode_data.get("chunk_size", 30)
    task_name = episode_data.get("task_name", "unknown")
    episode_idx = episode_data.get("episode_index", episode_data.get("demo_key", "?"))

    fig, ax = plt.subplots(figsize=(16, 5))

    # --- Ground truth line ---
    T = len(gt_actions)
    t_axis = np.arange(T)
    ax.plot(
        t_axis,
        gt_actions[:, dim],
        color=GROUND_TRUTH_COLOR,
        linewidth=1.2,
        label="Ground truth (demonstration)",
        zorder=2,
    )

    # --- Predicted trajectories ---
    plotted_pred_label = False
    for pred in predictions:
        start_t = pred["start_t"]
        pred_actions = np.array(pred["predicted_actions"])  # (chunk_size, A)
        pred_len = len(pred_actions)
        pred_t = np.arange(start_t, start_t + pred_len)

        # Clip to episode bounds
        mask = pred_t < T
        pred_t = pred_t[mask]
        pred_vals = pred_actions[: len(pred_t), dim]

        if len(pred_t) == 0:
            continue

        label = "X-VLA prediction (30 steps)" if not plotted_pred_label else None
        ax.plot(
            pred_t,
            pred_vals,
            color=PREDICTION_COLOR,
            alpha=PREDICTION_ALPHA,
            linewidth=2.0,
            label=label,
            zorder=1,
        )
        plotted_pred_label = True

        # Mark prediction start on the predicted trajectory
        if show_prediction_start:
            ax.scatter(
                pred_t[0],
                pred_vals[0],
                color=PREDICTION_COLOR,
                s=MARKER_SIZE,
                marker=PRED_MARKER,
                zorder=5,
                edgecolors="white",
                linewidths=0.5,
            )

    # --- Mark prediction starts on ground truth ---
    if show_prediction_start:
        for pred in predictions:
            start_t = pred["start_t"]
            if start_t < T:
                ax.scatter(
                    start_t,
                    gt_actions[start_t, dim],
                    color=GROUND_TRUTH_COLOR,
                    s=MARKER_SIZE,
                    marker=GT_MARKER,
                    zorder=5,
                    edgecolors="white",
                    linewidths=0.5,
                )

    # --- Styling ---
    dim_label = DIM_LABELS[dim] if dim < len(DIM_LABELS) else f"dim_{dim}"
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel(dim_label, fontsize=12)
    ax.set_title(
        f"{task_name} — {dim_label}\n(Episode {episode_idx}, {T} steps, predict every 10 steps, {chunk_size}-step chunks)",
        fontsize=11,
        fontweight="normal",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, T)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_task_combined_plot(
    task_data: dict,
    dim: int,
    output_path: str,
):
    """
    Create a single plot showing all episodes for a task overlaid,
    with ground truth as solid lines and predicted chunks overlaid.

    For tasks with many episodes, this plots each episode's ground truth
    as a separate thin line and overlays predictions.
    """
    episodes = task_data.get("episodes", [])
    if not episodes:
        print(f"  No episodes in task data, skipping.")
        return

    chunk_size = task_data.get("chunk_size", 30)
    task_name = task_data.get("task_name", "unknown")
    dim_label = DIM_LABELS[dim] if dim < len(DIM_LABELS) else f"dim_{dim}"

    n_episodes = len(episodes)
    n_cols = min(3, n_episodes)
    n_rows = int(np.ceil(n_episodes / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for ep_i, ep_data in enumerate(episodes):
        row, col = divmod(ep_i, n_cols)
        ax = axes[row][col]

        gt_actions = np.array(ep_data["ground_truth_actions"])
        T = len(gt_actions)
        t_axis = np.arange(T)

        # Ground truth
        ax.plot(
            t_axis, gt_actions[:, dim],
            color=GROUND_TRUTH_COLOR, linewidth=1.0,
            label="Ground truth",
        )

        # Predictions
        plotted_pred = False
        for pred in ep_data.get("predictions", []):
            start_t = pred["start_t"]
            pred_actions = np.array(pred["predicted_actions"])
            pred_len = len(pred_actions)
            pred_t = np.arange(start_t, start_t + pred_len)
            mask = pred_t < T
            pred_t = pred_t[mask]
            pred_vals = pred_actions[: len(pred_t), dim]

            if len(pred_t) == 0:
                continue

            lbl = "X-VLA prediction" if not plotted_pred else None
            ax.plot(pred_t, pred_vals, color=PREDICTION_COLOR,
                    alpha=PREDICTION_ALPHA, linewidth=1.5, label=lbl)
            plotted_pred = True

            # Start marker on ground truth
            if start_t < T:
                ax.scatter(start_t, gt_actions[start_t, dim],
                          color=GROUND_TRUTH_COLOR, s=MARKER_SIZE, marker=GT_MARKER,
                          zorder=5, edgecolors="white", linewidths=0.5)
            # Start marker on prediction
            ax.scatter(pred_t[0], pred_vals[0],
                      color=PREDICTION_COLOR, s=MARKER_SIZE, marker=PRED_MARKER,
                      zorder=5, edgecolors="white", linewidths=0.5)

        ax.set_xlabel("Timestep", fontsize=9)
        ax.set_ylabel(dim_label, fontsize=9)
        ep_label = ep_data.get('episode_index', ep_data.get('demo_key', '?'))
        ax.set_title(f"Ep {ep_label} ({T} steps)", fontsize=10)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(0, T)

    # Hide unused subplots
    for ep_i in range(n_episodes, n_rows * n_cols):
        row, col = divmod(ep_i, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"{task_name} — {dim_label}\n"
        f"({n_episodes} episodes, {chunk_size}-step prediction chunks every 10 steps)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot X-VLA predictions vs ground truth per dimension."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="output/raw_data",
        help="Directory containing raw JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Root output directory (will create dim_*/ subdirs).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Plot only this specific dimension (0-6).",
    )
    parser.add_argument(
        "--single-episode",
        type=int,
        default=None,
        help="Plot only a single episode index (0-based, across all JSONs).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files in {input_dir}")

    dims_to_plot = [args.dim] if args.dim is not None else list(range(7))

    for json_path in json_files:
        with open(json_path) as f:
            task_data = json.load(f)

        task_name = task_data.get("task_name", json_path.stem)
        print(f"\nPlotting: {task_name}")

        for dim in dims_to_plot:
            dim_label = DIM_LABELS[dim] if dim < len(DIM_LABELS) else f"dim_{dim}"
            output_path = output_dir / f"dim_{dim}" / f"{json_path.stem}.png"

            make_task_combined_plot(task_data, dim, str(output_path))
            print(f"  dim_{dim} → {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
