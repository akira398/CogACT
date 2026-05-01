"""
replay_gt.py — replay GT actions from the LeRobot dataset in simulation.

Uses PandaMobile (the robot the demos were recorded with) so the actions
execute correctly.  Each episode's exact scene (layout_id, style_id) is
read from the per-episode ep_meta.json, ensuring the kitchen matches the
recorded actions.

Usage:
    # Single task, 3 episodes, each in its original scene:
    python scripts/replay_gt.py \
        --gt_data_root datasets/robocasa/v1.0/target \
        --tasks TurnOnMicrowave \
        --n_episodes 3 \
        --output_dir results/gt_replay

    # Verbose (print step rewards):
    python scripts/replay_gt.py \
        --gt_data_root datasets/robocasa/v1.0/target \
        --tasks TurnOnMicrowave \
        --verbose \
        --output_dir results/gt_replay
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Episode loading ────────────────────────────────────────────────────────────

def _find_ep_meta(task_ds: Path, ep_idx: int) -> dict:
    """Return the ep_meta dict for episode ep_idx, or {} if not found."""
    # LeRobot extras layout: <task_ds>/<date>/lerobot/extras/episode_XXXXXX/ep_meta.json
    # or directly: <task_ds>/extras/episode_XXXXXX/ep_meta.json
    ep_dir_name = f"episode_{ep_idx:06d}"
    for meta_path in task_ds.rglob(f"{ep_dir_name}/ep_meta.json"):
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _load_episodes(gt_data_root: Path, task_name: str, n_episodes: int):
    """Return list of (ep_idx, actions, ep_meta) for up to n_episodes."""
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
        print("  [ERROR] pandas required: pip install pandas pyarrow")
        return []

    dfs = []
    for f in parquet_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}")
    if not dfs:
        return []
    df = pd.concat(dfs, ignore_index=True)

    action_col = next((c for c in ("action", "actions") if c in df.columns), None)
    action_dim_cols = sorted(c for c in df.columns if c.startswith("action."))

    if action_col is None and not action_dim_cols:
        print(f"  [ERROR] No action column. Available: {list(df.columns)[:15]}")
        return []

    ep_indices = sorted(df["episode_index"].unique()) if "episode_index" in df.columns else [0]

    episodes = []
    for ep_idx in ep_indices[:n_episodes]:
        ep_df = df[df["episode_index"] == ep_idx].reset_index(drop=True) if "episode_index" in df.columns else df

        if action_col:
            actions = np.stack(ep_df[action_col].to_list()).astype(np.float32)
        else:
            actions = ep_df[action_dim_cols].values.astype(np.float32)

        ep_meta = _find_ep_meta(task_ds, int(ep_idx))
        episodes.append((int(ep_idx), actions, ep_meta))

    action_dim = episodes[0][1].shape[1] if episodes else "?"
    print(f"  Loaded {len(episodes)} episodes  action_dim={action_dim}")

    missing_meta = sum(1 for _, _, m in episodes if not m)
    if missing_meta:
        print(f"  [WARN] {missing_meta}/{len(episodes)} episodes have no ep_meta.json "
              f"(will use --layout_id/--style_id fallback)")

    return episodes


# ── Replay ─────────────────────────────────────────────────────────────────────

def _print_action_info(env, gt_actions: np.ndarray) -> None:
    """Print action split and a few GT samples so we can diagnose ordering issues."""
    low, high = env.action_spec
    print(f"    env action_dim={len(low)}")

    # Print composite controller action split if available.
    try:
        robot = env.robots[0]
        cc = robot.composite_controller
        print(f"    composite_controller action split:")
        for part, (s, e) in cc._action_split_indexes.items():
            print(f"      [{s}:{e}] ({e-s} dim) → {part}")
    except Exception as e:
        print(f"    (could not read action split: {e})")

    print(f"    GT action_dim={gt_actions.shape[1]}")
    print(f"    First 3 GT actions:")
    for i, a in enumerate(gt_actions[:3]):
        print(f"      [{i}] {np.array2string(a, precision=3, suppress_small=True)}")


def replay_episode(env, actions: np.ndarray, camera_name: str,
                   horizon: int, verbose: bool, diag: bool = False):
    """Replay actions in env; return (success, frames)."""
    obs = env.reset()
    frames = []
    action_dim = env.action_spec[0].shape[0]

    if diag:
        _print_action_info(env, actions)

    if actions.shape[1] != action_dim:
        print(f"    [WARN] GT action_dim={actions.shape[1]}, env action_dim={action_dim} "
              f"— truncating/padding.")

    success = False
    for step in range(min(len(actions), horizon)):
        img_np = obs[f"{camera_name}_image"]
        if img_np.ndim == 4:
            img_np = img_np[0]
        frames.append(img_np.copy())

        a = actions[step]
        if len(a) >= action_dim:
            a = a[:action_dim]
        else:
            a = np.concatenate([a, np.zeros(action_dim - len(a))])

        obs, reward, done, info = env.step(a)

        if verbose and step % 20 == 0:
            print(f"      step {step:4d}  reward={reward:.4f}")

        if done:
            success = bool(info.get("success", False))
            if verbose:
                print(f"      done at step {step}  success={success}")
            break
    else:
        try:
            success = bool(env._check_success())
        except Exception:
            pass

    return success, frames


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--gt_data_root", required=True,
                   help="LeRobot dataset root (e.g. datasets/robocasa/v1.0/target)")
    p.add_argument("--tasks", nargs="+", default=["TurnOnMicrowave"])
    p.add_argument("--n_episodes", type=int, default=3,
                   help="GT episodes to replay per task.")
    p.add_argument("--robot", default="PandaMobile",
                   help="Robot. PandaMobile matches the GT recordings.")
    p.add_argument("--controller", default="OSC_POSE")
    p.add_argument("--camera_name", default="robot0_agentview_left")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--object_instance_split", default="target")
    p.add_argument("--horizon", type=int, default=1500,
                   help="Max steps per episode.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--output_dir", default="results/gt_replay")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--diag", action="store_true",
                   help="Print action split and first GT actions on episode 0 to diagnose ordering.")
    # Fallback scene if ep_meta.json is missing:
    p.add_argument("--layout_id", type=int, default=1)
    p.add_argument("--style_id", type=int, default=1)
    args = p.parse_args()

    gt_root = Path(args.gt_data_root)
    if not gt_root.exists():
        print(f"ERROR: {gt_root} does not exist", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.eval_robocasa365 import make_env, save_video

    summary = []

    for task_name in args.tasks:
        print(f"\n{'='*60}")
        print(f"  Task: {task_name}  robot={args.robot}")
        print(f"{'='*60}")

        episodes = _load_episodes(gt_root, task_name, args.n_episodes)
        if not episodes:
            continue

        for ep_idx, actions, ep_meta in episodes:
            layout_id = ep_meta.get("layout_id", args.layout_id)
            style_id  = ep_meta.get("style_id",  args.style_id)
            scene_src = "ep_meta" if ep_meta else "fallback"
            label = f"{task_name}_layout{layout_id}_style{style_id}_ep{ep_idx:03d}"
            out_path = out_dir / f"{label}_GT.mp4"

            print(f"\n  Episode {ep_idx}: {len(actions)} steps  "
                  f"scene=layout{layout_id}/style{style_id} ({scene_src})")
            print(f"  → {out_path.name}")

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
                print(f"  [ERROR] env creation failed: {e}")
                import traceback; traceback.print_exc()
                summary.append((task_name, ep_idx, "ENV_ERROR"))
                continue

            first_ep = (ep_idx == episodes[0][0])
            try:
                success, frames = replay_episode(
                    env, actions,
                    camera_name=args.camera_name,
                    horizon=args.horizon,
                    verbose=args.verbose,
                    diag=(args.diag and first_ep),
                )
                env.close()
            except Exception as e:
                print(f"  [ERROR] replay failed: {e}")
                import traceback; traceback.print_exc()
                try:
                    env.close()
                except Exception:
                    pass
                summary.append((task_name, ep_idx, "REPLAY_ERROR"))
                continue

            status = "SUCCESS" if success else "FAIL"
            print(f"  {status}  ({len(frames)} frames)")
            summary.append((task_name, ep_idx, status))

            if frames:
                save_video(frames, out_path, fps=args.fps)
                print(f"  Saved: {out_path}")

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for task_name, ep_idx, status in summary:
        print(f"  {task_name}  ep{ep_idx:03d}  {status}")
    n_ok = sum(1 for _, _, s in summary if s == "SUCCESS")
    print(f"\n  {n_ok}/{len(summary)} succeeded")
    print(f"\nVideos: {out_dir}/")


if __name__ == "__main__":
    main()
