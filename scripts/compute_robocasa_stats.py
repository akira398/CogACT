"""
compute_robocasa_stats.py

Compute action normalization statistics (q01, q99) from RoboCasa HDF5 or
LeRobot Parquet demos and save them as dataset_statistics.json — the format
expected by CogACT's predict_action().

RoboCasa365 v1.0 uses LeRobot Parquet format.
RoboCasa v0.2 used HDF5 (robomimic) format.
This script handles both automatically.

Usage:
    # RoboCasa365 v1.0 (Parquet):
    python scripts/compute_robocasa_stats.py \
        --data_root datasets/robocasa/v1.0/target \
        --output_path data/robocasa/dataset_statistics.json

    # Subset of tasks (faster approximation):
    python scripts/compute_robocasa_stats.py \
        --data_root datasets/robocasa/v1.0/target \
        --tasks PickPlaceCounterToCabinet TurnOnMicrowave \
        --output_path data/robocasa/dataset_statistics.json

    # Old RoboCasa v0.2 HDF5 data:
    python scripts/compute_robocasa_stats.py \
        --data_root data/robocasa_v02 \
        --output_path data/robocasa/dataset_statistics.json

    Then pass to eval:
    python scripts/eval_robocasa365.py \
        --model_path pretrained/CogACT-Base \
        --norm_stats_path data/robocasa/dataset_statistics.json \
        --unnorm_key robocasa
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute RoboCasa action normalization statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", type=str, required=True,
                   help="Directory containing .parquet or .hdf5 files (searched recursively).")
    p.add_argument("--output_path", type=str, default=None,
                   help="Where to save dataset_statistics.json. "
                        "Defaults to <data_root>/dataset_statistics.json.")
    p.add_argument("--tasks", type=str, nargs="*", default=None,
                   help="If given, only load files whose path contains one of these task names.")
    p.add_argument("--action_dim", type=int, default=7,
                   help="Expected action dimensionality (7 for CogACT).")
    p.add_argument("--quantile_low", type=float, default=0.01)
    p.add_argument("--quantile_high", type=float, default=0.99)
    p.add_argument("--key", type=str, default="robocasa",
                   help="Key under which to store stats in dataset_statistics.json.")
    return p.parse_args()


def collect_actions_parquet(data_root: Path, tasks: Optional[List[str]]) -> np.ndarray:
    """Read actions from LeRobot Parquet files (RoboCasa365 v1.0)."""
    import pandas as pd

    parquet_files = sorted(data_root.rglob("*.parquet"))
    if tasks:
        parquet_files = [f for f in parquet_files
                         if any(task in str(f) for task in tasks)]
    assert parquet_files, f"No .parquet files found under {data_root}"
    print(f"Found {len(parquet_files)} Parquet file(s)")

    all_actions = []
    for pq_path in tqdm(parquet_files, desc="Reading Parquet files"):
        try:
            df = pd.read_parquet(pq_path)
        except Exception as e:
            print(f"  [WARN] Could not read {pq_path}: {e}")
            continue

        # Find the action column — LeRobot uses "action"; older conversions may use "actions"
        action_col = None
        for candidate in ("action", "actions"):
            if candidate in df.columns:
                action_col = candidate
                break
        if action_col is None:
            print(f"  [WARN] No 'action'/'actions' column in {pq_path.name}, skipping. "
                  f"Columns: {list(df.columns)[:10]}")
            continue

        raw = df[action_col]
        # Each cell is a list/array of floats (one per DoF)
        try:
            actions = np.stack(raw.to_list()).astype(np.float32)
        except Exception as e:
            print(f"  [WARN] Could not stack actions from {pq_path.name}: {e}")
            continue

        # Filter to task subset using task_index / task columns if present
        if tasks and "task" in df.columns:
            mask = df["task"].apply(lambda t: any(task in str(t) for task in tasks))
            actions = actions[mask.to_numpy()]

        if actions.shape[0] > 0:
            all_actions.append(actions)

    assert all_actions, (
        "No actions found in any Parquet file! "
        "Check that the action column exists and is non-empty."
    )
    return np.concatenate(all_actions, axis=0)


def collect_actions_hdf5(data_root: Path, tasks: Optional[List[str]]) -> np.ndarray:
    """Read actions from robomimic HDF5 files (RoboCasa v0.2)."""
    import h5py

    hdf5_files = sorted(data_root.rglob("*.hdf5"))
    if tasks:
        hdf5_files = [f for f in hdf5_files if f.parent.name in tasks]
    assert hdf5_files, f"No .hdf5 files found under {data_root}"
    print(f"Found {len(hdf5_files)} HDF5 file(s)")

    all_actions = []
    for hdf5_path in tqdm(hdf5_files, desc="Reading HDF5 files"):
        with h5py.File(str(hdf5_path), "r") as f:
            if "data" not in f:
                continue
            for demo_key in f["data"]:
                demo = f["data"][demo_key]
                if "actions" not in demo:
                    continue
                all_actions.append(demo["actions"][:])

    assert all_actions, "No actions found in any HDF5 file!"
    return np.concatenate(all_actions, axis=0)


def collect_actions(data_root: Path, tasks: Optional[List[str]]) -> np.ndarray:
    """Auto-detect format and collect actions."""
    parquet_files = list(data_root.rglob("*.parquet"))
    hdf5_files = list(data_root.rglob("*.hdf5"))

    if parquet_files and not hdf5_files:
        print("Detected LeRobot Parquet format (RoboCasa365 v1.0)")
        return collect_actions_parquet(data_root, tasks)
    elif hdf5_files and not parquet_files:
        print("Detected HDF5 format (RoboCasa v0.2)")
        return collect_actions_hdf5(data_root, tasks)
    elif parquet_files and hdf5_files:
        print(f"Found both Parquet ({len(parquet_files)}) and HDF5 ({len(hdf5_files)}) files — "
              "using Parquet (v1.0).")
        return collect_actions_parquet(data_root, tasks)
    else:
        raise FileNotFoundError(
            f"No .parquet or .hdf5 files found under {data_root}\n"
            "Make sure you ran:\n"
            '  echo "y" | python -m robocasa.scripts.download_datasets '
            "--split target --source human"
        )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    print(f"Collecting actions from: {data_root}")
    if args.tasks:
        print(f"  Filtering to tasks: {args.tasks}")

    all_actions = collect_actions(data_root, args.tasks)
    print(f"Total transitions collected: {len(all_actions):,}")
    print(f"Action shape: {all_actions.shape}")

    if all_actions.shape[1] != args.action_dim:
        print(
            f"[WARN] Found action_dim={all_actions.shape[1]}, expected {args.action_dim}. "
            "Proceeding anyway."
        )

    q01 = np.quantile(all_actions, args.quantile_low, axis=0)
    q99 = np.quantile(all_actions, args.quantile_high, axis=0)

    print(f"\nAction statistics (q{args.quantile_low:.0%} / q{args.quantile_high:.0%}):")
    dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    for dim_i in range(all_actions.shape[1]):
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
