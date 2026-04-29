"""
compute_robocasa_stats.py

Compute action normalization statistics (q01, q99) from RoboCasa HDF5 demos and
save them as dataset_statistics.json — the format expected by CogACT's predict_action().

This is required for zero-shot evaluation with a pretrained CogACT model, because the
model's built-in norm_stats are from Open-X Embodiment, not RoboCasa.

Usage:
    # Compute from downloaded HDF5 data:
    python scripts/compute_robocasa_stats.py \
        --data_root data/robocasa \
        --output_path data/robocasa/dataset_statistics.json

    # Or compute from a subset of tasks for quick approximation:
    python scripts/compute_robocasa_stats.py \
        --data_root data/robocasa \
        --tasks PickPlaceCounterToCabinet TurnOnMicrowave \
        --output_path data/robocasa/dataset_statistics.json

    Then pass to eval:
    python scripts/eval_robocasa365.py \
        --model_path CogACT/CogACT-Base \
        --norm_stats_path data/robocasa/dataset_statistics.json \
        --unnorm_key robocasa
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute RoboCasa action normalization statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", type=str, required=True,
                   help="Directory containing .hdf5 files (searched recursively).")
    p.add_argument("--output_path", type=str, default=None,
                   help="Where to save dataset_statistics.json. "
                        "Defaults to <data_root>/dataset_statistics.json.")
    p.add_argument("--tasks", type=str, nargs="*", default=None,
                   help="If given, only load HDF5 files whose parent directory name "
                        "matches one of these task names.")
    p.add_argument("--action_dim", type=int, default=7,
                   help="Expected action dimensionality (7 for CogACT).")
    p.add_argument("--quantile_low", type=float, default=0.01)
    p.add_argument("--quantile_high", type=float, default=0.99)
    p.add_argument("--key", type=str, default="robocasa",
                   help="Key under which to store stats in dataset_statistics.json.")
    return p.parse_args()


def collect_actions(data_root: Path, tasks: Optional[List[str]]) -> np.ndarray:
    hdf5_files = sorted(data_root.rglob("*.hdf5"))
    if tasks:
        hdf5_files = [f for f in hdf5_files if f.parent.name in tasks]
    assert hdf5_files, f"No .hdf5 files found under {data_root}"

    all_actions = []
    for hdf5_path in tqdm(hdf5_files, desc="Reading HDF5 files"):
        with h5py.File(str(hdf5_path), "r") as f:
            if "data" not in f:
                continue
            for demo_key in f["data"]:
                demo = f["data"][demo_key]
                if "actions" not in demo:
                    continue
                actions = demo["actions"][:]
                all_actions.append(actions)

    assert all_actions, "No actions found!"
    return np.concatenate(all_actions, axis=0)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    print(f"Collecting actions from: {data_root}")
    if args.tasks:
        print(f"  Filtering to tasks: {args.tasks}")

    all_actions = collect_actions(data_root, args.tasks)
    print(f"Total transitions collected: {len(all_actions):,}")
    print(f"Action shape: {all_actions.shape}")

    # Verify action dim
    if all_actions.shape[1] != args.action_dim:
        print(
            f"[WARN] Found action_dim={all_actions.shape[1]}, expected {args.action_dim}. "
            "Proceeding anyway."
        )

    q01 = np.quantile(all_actions, args.quantile_low, axis=0)
    q99 = np.quantile(all_actions, args.quantile_high, axis=0)

    print(f"\nAction statistics (q{args.quantile_low:.0%} / q{args.quantile_high:.0%}):")
    for dim_i in range(all_actions.shape[1]):
        dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        name = dim_names[dim_i] if dim_i < len(dim_names) else f"dim{dim_i}"
        print(f"  {name:8s}  q01={q01[dim_i]:+.4f}  q99={q99[dim_i]:+.4f}  "
              f"mean={all_actions[:, dim_i].mean():+.4f}  "
              f"std={all_actions[:, dim_i].std():.4f}")

    stats = {
        args.key: {
            "action": {
                "q01": q01.tolist(),
                "q99": q99.tolist(),
                "mask": [True] * all_actions.shape[1],
            }
        }
    }

    out_path = Path(args.output_path) if args.output_path else data_root / "dataset_statistics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to: {out_path}")
    print(f"Use with: --norm_stats_path {out_path} --unnorm_key {args.key}")


if __name__ == "__main__":
    main()
