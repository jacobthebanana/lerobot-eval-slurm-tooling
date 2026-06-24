#!/usr/bin/env python3
"""
test_xvla_google_robot.py — Verify XVLA google-robot model works with preprocessing.

Tests:
1. Model loads successfully
2. Preprocessor processes observation dict correctly
3. Model produces valid 20D actions
4. Action conversion (20D → 7D) works

This does NOT require SimplerEnv. Run on any node.

Usage:
    python test_xvla_google_robot.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

MODEL_ID = "lerobot/xvla-google-robot"


def main():
    print("=" * 60)
    print("Test: XVLA Google Robot Model")
    print("=" * 60)

    # 1. Load model
    print("\n1. Loading model...")
    start = time.time()
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = XVLAPolicy.from_pretrained(MODEL_ID).to(device).eval()
    print(f"   Loaded in {time.time() - start:.1f}s")
    print(f"   Device: {device}")
    print(f"   Action mode: {policy.config.action_mode}")
    print(f"   Chunk size: {policy.config.chunk_size}")
    print(f"   Denoising steps: {policy.config.num_denoising_steps}")
    print(f"   State dim: {policy.config.robot_state_feature.shape}")
    print(f"   Image dims: {[v.shape for k, v in policy.config.image_features.items()]}")

    # 2. Load preprocessor
    print("\n2. Loading preprocessor...")
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(f"   Steps: {[s.__class__.__name__ for s in preprocessor.steps]}")

    # 3. Create dummy observation
    print("\n3. Testing with dummy observation...")
    obs = {
        "observation.images.image": torch.rand(3, 256, 256, dtype=torch.float32),
        "observation.images.image2": torch.rand(3, 256, 256, dtype=torch.float32),
        "observation.images.empty_camera_0": torch.zeros(3, 224, 224, dtype=torch.float32),
        "observation.state": torch.rand(8, dtype=torch.float32),
        "task": "pick up the coke can",
    }
    print(f"   Input obs keys: {list(obs.keys())}")

    # 4. Preprocess
    batch = preprocessor(obs)
    print(f"\n4. Preprocessed batch keys: {list(batch.keys())}")

    # Move to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # 5. Run inference
    print("\n5. Running predict_action_chunk...")
    with torch.inference_mode():
        model_output = policy.predict_action_chunk(batch)
    print(f"   Output shape: {model_output.shape}")  # (1, 30, 20)
    print(f"   Output dtype: {model_output.dtype}")

    # 6. Check action statistics
    act_np = model_output[0].cpu().numpy()  # (30, 20)
    print(f"\n6. Action statistics (first chunk step, 30 steps total):")
    print(f"   First step (20 dims): {act_np[0].tolist()}")
    print(f"   First 7 dims over 30 steps:")
    print(f"     dx:   mean={act_np[:, 0].mean():.4f}, std={act_np[:, 0].std():.4f}")
    print(f"     dy:   mean={act_np[:, 1].mean():.4f}, std={act_np[:, 1].std():.4f}")
    print(f"     dz:   mean={act_np[:, 2].mean():.4f}, std={act_np[:, 2].std():.4f}")
    print(f"     rot6d: mean={act_np[:, 3:9].mean():.4f}, std={act_np[:, 3:9].std():.4f}")
    print(f"     gripper: mean={act_np[:, 9].mean():.4f}, std={act_np[:, 9].std():.4f}")

    # 7. Test select_action
    print("\n7. Testing select_action (closed-loop mock)...")
    policy.reset()
    with torch.inference_mode():
        action_20d = policy.select_action(batch)
    print(f"   select_action output shape: {action_20d.shape}")  # (1, 20)

    # 8. Test rotation conversion
    print("\n8. Testing 20D → 7D action conversion...")
    from eval_xvla_google_robot import convert_20d_to_7d

    act_7d = convert_20d_to_7d(act_np)  # (30, 7)
    print(f"   Converted shape: {act_7d.shape}")
    print(f"   First action 7D: {act_7d[0].tolist()}")
    print(f"   Gripper range: [{act_7d[:, 6].min():.1f}, {act_7d[:, 6].max():.1f}]")

    # 9. Verify rotation conversion is valid
    print("\n9. Validating action output...")
    assert model_output.shape == (1, 30, 20), f"Expected (1, 30, 20), got {model_output.shape}"
    assert act_7d.shape == (30, 7), f"Expected (30, 7), got {act_7d.shape}"
    assert np.all(np.abs(act_7d[:, 6]) == 1.0), "Gripper should be binary ±1"
    print(f"   All assertions passed!")

    print(f"\n{'=' * 60}")
    print("TEST PASSED — Model loads, preprocesses, and infers correctly")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
