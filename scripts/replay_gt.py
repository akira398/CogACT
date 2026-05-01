"""
replay_gt.py — replay GT actions from the LeRobot dataset in simulation.

Uses PandaMobile (the robot the demos were recorded with) so the actions
execute correctly.  Useful as a sanity check: if GT replay fails, the
environment setup (robot placement, cameras, scene) is broken.

Usage:
    python scripts/replay_gt.py \
        --gt_data_root datasets/robocasa/v1.0/target \
        --tasks TurnOnMicrowave \
        --output_dir results/gt_replay

    # Try all 5 eval scenes, 2 episodes each:
    python scripts/replay_gt.py \
        --gt_data_root datasets/robocasa/v1.0/target \
        --tasks TurnOnMicrowave \
        --all_scenes \
        --n_episodes 2 \
        --output_dir results/gt_replay

    # Verbose step-by-step (see rewards each step):
    python scripts/replay_gt.py \
        --gt_data_root datasets/robocasa/v1.0/target \
        --tasks TurnOnMicrowave \
        --verbose \
        --output_dir results/gt_replay
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np

# Bring eval_robocasa365 into path so we can import shared helpers.
sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_episodes(gt_data_root: Path, task_name: str, n_episodes: int):
    """Return a list of (T, action_dim) arrays, one per episode (up to n_episodes)."""
    from scripts.eval_robocasa365 import _find_task_dir

    task_ds = _find_task_dir(gt_data_root, task_name)
    if task_ds is None:
        print(f"  [ERROR] Task '{task_name}' not found under {gt_data_root}")
        return []

    parquet_files = sorted(task_ds.rglob("*.parquet"))
    if not parquet_files:
        print(f"  [ERROR] No parquet files found for {task_name} in {task_ds}")
        return []

    try:
        import pandas as pd
    except ImportError:
        print("  [ERROR] pandas is required: pip install pandas pyarrow")
        return []

    # Collect all rows from all parquet files and group by episode_index.
    dfs = []
    for f in parquet_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}")
    if not dfs:
        return []
    df = pd.concat(dfs, ignore_index=True)

    # Detect action columns.
    action_col = next((c for c in ("action", "actions") if c in df.columns), None)
    action_dim_cols = sorted(c for c in df.columns if c.startswith("action."))

    if action_col is None and not action_dim_cols:
        print(f"  [ERROR] No action column found. Available: {list(df.columns)[:15]}")
        return []

    episodes = []
    ep_indices = sorted(df["episode_index"].unique()) if "episode_index" in df.columns else [None]

    for ep_idx in ep_indices[:n_episodes]:
        if ep_idx is not None:
            ep_df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
        else:
            ep_df = df

        if action_col:
            vals = ep_df[action_col].to_list()
            actions = np.stack(vals).astype(np.float32)
        else:
            actions = ep_df[action_dim_cols].values.astype(np.float32)

        episodes.append((int(ep_idx) if ep_idx is not None else 0, actions))

    print(f"  Loaded {len(episodes)} episodes for {task_name}  "
          f"(action_dim={episodes[0][1].shape[1] if episodes else '?'})")
    return episodes


def replay_episode(env, actions, camera_name: str, horizon: int, verbose: bool):
    """Replay actions in env; return (success, frames)."""
    obs = env.reset()
    frames = []
    success = False
    action_dim = env.action_spec[0].shape[0]

    if actions.shape[1] != action_dim:
        print(f"    [WARN] GT action_dim={actions.shape[1]}, env action_dim={action_dim}. "
              f"Truncating/padding GT actions to {action_dim}.")

    for step in range(min(len(actions), horizon)):
        img_np = obs[f"{camera_name}_image"]
        if img_np.ndim == 4:
            img_np = img_np[0]
        frames.append(img_np.copy())

        # Adapt action dimension.
        a = actions[step]
        if len(a) >= action_dim:
            a = a[:action_dim]
        else:
            a = np.concatenate([a, np.zeros(action_dim - len(a))])

        obs, reward, done, info = env.step(a)

        if verbose and step % 20 == 0:
            print(f"      step {step:4d}  reward={reward:.3f}")

        if done:
            success = bool(info.get("success", False))
            if verbose:
                print(f"      done at step {step}  success={success}")
            break
    else:
        # Horizon reached — check success from last info.
        try:
            success = bool(env._check_success())
        except Exception:
            pass

    return success, frames


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--gt_data_root", required=True,
                   help="Path to LeRobot dataset root (e.g. datasets/robocasa/v1.0/target)")
    p.add_argument("--tasks", nargs="+", default=["TurnOnMicrowave"],
                   help="Task names to replay.")
    p.add_argument("--robot", default="PandaMobile",
                   help="Robot type. Use PandaMobile to match GT demos.")
    p.add_argument("--controller", default="OSC_POSE")
    p.add_argument("--camera_name", default="robot0_agentview_left")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--layout_id", type=int, default=1,
                   help="Kitchen layout to use (0–4). Ignored if --all_scenes.")
    p.add_argument("--style_id", type=int, default=1,
                   help="Kitchen style to use (0–4). Ignored if --all_scenes.")
    p.add_argument("--all_scenes", action="store_true",
                   help="Cycle through all 5 eval (layout, style) pairs instead of one scene.")
    p.add_argument("--n_episodes", type=int, default=1,
                   help="Number of GT episodes to replay per task (per scene).")
    p.add_argument("--horizon", type=int, default=1000,
                   help="Max steps per episode.")
    p.add_argument("--object_instance_split", default="target")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--output_dir", default="results/gt_replay")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    gt_root = Path(args.gt_data_root)
    if not gt_root.exists():
        print(f"ERROR: --gt_data_root does not exist: {gt_root}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import shared helpers (also applies all monkey-patches via make_env).
    from scripts.eval_robocasa365 import make_env, save_video, EVAL_SCENES

    scenes = EVAL_SCENES if args.all_scenes else [(args.layout_id, args.style_id)]

    for task_name in args.tasks:
        print(f"\n{'='*60}")
        print(f"  Task: {task_name}  robot={args.robot}")
        print(f"{'='*60}")

        episodes = _load_episodes(gt_root, task_name, args.n_episodes)
        if not episodes:
            continue

        for layout_id, style_id in scenes:
            scene_label = f"layout{layout_id}_style{style_id}"
            print(f"\n  Scene: {scene_label}")

            for ep_idx, actions in episodes:
                label = f"{task_name}_{scene_label}_ep{ep_idx:03d}"
                out_path = out_dir / f"{label}_GT.mp4"

                print(f"    Episode {ep_idx}: {len(actions)} steps → {out_path.name}")

                try:
                    env = make_env(
                        task_name=task_name,
                        layout_id=layout_id,
                        style_id=style_id,
                        robot=args.robot,
                        controller=args.controller,
                        camera_name=args.camera_name,
                        img_size=args.img_size,
                        object_instance_split=args.object_instance_split,
                    )
                except Exception as e:
                    print(f"    [ERROR] env creation failed: {e}")
                    import traceback; traceback.print_exc()
                    continue

                try:
                    success, frames = replay_episode(
                        env, actions,
                        camera_name=args.camera_name,
                        horizon=args.horizon,
                        verbose=args.verbose,
                    )
                    env.close()
                except Exception as e:
                    print(f"    [ERROR] replay failed: {e}")
                    import traceback; traceback.print_exc()
                    try:
                        env.close()
                    except Exception:
                        pass
                    continue

                status = "SUCCESS" if success else "FAIL"
                print(f"    {status}  ({len(frames)} frames recorded)")
                if frames:
                    save_video(frames, out_path, fps=args.fps)
                    print(f"    Saved: {out_path}")

    print(f"\nDone. Videos in {out_dir}/")


if __name__ == "__main__":
    main()
