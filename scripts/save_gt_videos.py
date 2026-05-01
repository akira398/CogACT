"""
save_gt_videos.py — copy pre-rendered GT demo videos from the LeRobot dataset.

The RoboCasa365 LeRobot format stores one MP4 per episode under:
  <gt_data_root>/{atomic,composite}/<Task>/videos/chunk-000/
    observation.images.<camera_name>/episode_XXXXXX.mp4

This script copies those MP4s to --output_dir without running any simulation.

Usage:
    # Single task, auto-detect camera, first 3 episodes
    python scripts/save_gt_videos.py \
        --gt_data_root datasets/robocasa/v1.0/target \
        --tasks TurnOnMicrowave \
        --n_episodes 3 \
        --output_dir results/gt_videos

    # List available tasks + cameras (no copy)
    python scripts/save_gt_videos.py \
        --gt_data_root datasets/robocasa/v1.0/target \
        --list

    # Specific camera
    python scripts/save_gt_videos.py \
        --gt_data_root datasets/robocasa/v1.0/target \
        --tasks TurnOnMicrowave PickPlaceCounterToCabinet \
        --camera agentview_left \
        --n_episodes 5 \
        --output_dir results/gt_videos
"""

import argparse
import shutil
import sys
from pathlib import Path


def find_task_dir(gt_data_root: Path, task_name: str):
    direct = gt_data_root / task_name
    if direct.exists():
        return direct
    for subdir in sorted(gt_data_root.iterdir()):
        if subdir.is_dir():
            candidate = subdir / task_name
            if candidate.exists():
                return candidate
    return None


def list_available(gt_data_root: Path) -> None:
    print(f"\nScanning {gt_data_root} ...\n")
    found = []
    for p in sorted(gt_data_root.rglob("videos")):
        task_dir = p.parent
        task_name = task_dir.name
        chunk = p / "chunk-000"
        cameras = [d.name for d in sorted(chunk.iterdir()) if d.is_dir()] if chunk.is_dir() else []
        n_eps = 0
        if cameras:
            n_eps = len(list((chunk / cameras[0]).glob("episode_*.mp4")))
        found.append((task_name, cameras, n_eps))

    if not found:
        print("  No LeRobot video directories found.")
        return

    print(f"{'Task':<45}  {'Episodes':>8}  Cameras")
    print("-" * 90)
    for name, cams, n_eps in found:
        cam_str = ", ".join(c.replace("observation.images.", "") for c in cams[:3])
        if len(cams) > 3:
            cam_str += f" (+{len(cams)-3} more)"
        print(f"  {name:<43}  {n_eps:>8}  {cam_str}")
    print(f"\n{len(found)} tasks found.")


def pick_camera_dir(vid_root: Path, camera_hint: str = None):
    if not vid_root.is_dir():
        return None
    cam_dirs = [d for d in sorted(vid_root.iterdir()) if d.is_dir()]
    if not cam_dirs:
        return None
    if camera_hint:
        for d in cam_dirs:
            if camera_hint in d.name:
                return d
    # Default priority: agentview_left > agentview > first available
    for pref in ("agentview_left", "agentview"):
        for d in cam_dirs:
            if pref in d.name:
                return d
    return cam_dirs[0]


def save_gt_videos(gt_data_root: Path, tasks: list, camera_hint: str,
                   n_episodes: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    any_saved = False

    for task_name in tasks:
        task_ds = find_task_dir(gt_data_root, task_name)
        if task_ds is None:
            print(f"  [SKIP] {task_name} — not found under {gt_data_root}")
            continue

        vid_root = task_ds / "videos" / "chunk-000"
        cam_dir = pick_camera_dir(vid_root, camera_hint)
        if cam_dir is None:
            print(f"  [SKIP] {task_name} — no video directories in {vid_root}")
            continue

        mp4s = sorted(cam_dir.glob("episode_*.mp4"))
        if not mp4s:
            print(f"  [SKIP] {task_name} — no episode MP4s in {cam_dir}")
            continue

        cam_label = cam_dir.name.replace("observation.images.", "")
        to_copy = mp4s[:n_episodes]
        print(f"\n  {task_name}  ({cam_label})  —  {len(to_copy)}/{len(mp4s)} episodes")

        for i, src in enumerate(to_copy):
            dst = output_dir / f"{task_name}_ep{i:03d}_GT.mp4"
            shutil.copy(src, dst)
            print(f"    [{i}] {dst.name}")
            any_saved = True

    if any_saved:
        print(f"\nVideos saved to {output_dir}/")
    else:
        print("\nNo videos were saved.")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--gt_data_root", type=str, required=True,
                   help="Path to downloaded LeRobot dataset root (e.g. datasets/robocasa/v1.0/target)")
    p.add_argument("--tasks", type=str, nargs="+", default=None,
                   help="Task names to copy videos for. Omit to copy all found tasks.")
    p.add_argument("--camera", type=str, default=None,
                   help="Camera name substring to match (e.g. 'agentview_left'). "
                        "Defaults to agentview_left if available.")
    p.add_argument("--n_episodes", type=int, default=3,
                   help="Number of episodes to copy per task.")
    p.add_argument("--output_dir", type=str, default="results/gt_videos")
    p.add_argument("--list", action="store_true",
                   help="List available tasks and cameras, then exit.")
    args = p.parse_args()

    gt_root = Path(args.gt_data_root)
    if not gt_root.exists():
        print(f"ERROR: --gt_data_root does not exist: {gt_root}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        list_available(gt_root)
        return

    if args.tasks:
        tasks = args.tasks
    else:
        # Collect all task dirs that have a videos/ subdir
        tasks = []
        for p in sorted(gt_root.rglob("videos")):
            name = p.parent.name
            if name not in tasks:
                tasks.append(name)
        if not tasks:
            print("No tasks found. Use --list to inspect the data root.")
            sys.exit(1)
        print(f"No --tasks given; found {len(tasks)} tasks.")

    save_gt_videos(gt_root, tasks, args.camera, args.n_episodes, Path(args.output_dir))


if __name__ == "__main__":
    main()
