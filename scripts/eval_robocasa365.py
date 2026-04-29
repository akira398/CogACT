"""
eval_robocasa365.py

Implements the exact RoboCasa365 evaluation protocol for CogACT.

Protocol (from RoboCasa365 paper + dataset_registry.py):
  - 50 tasks: 18 atomic-seen + 16 composite-seen + 16 composite-unseen
  - 5 fixed evaluation scenes (layout_id, style_id) x N trials each
  - Object instance split "B" (held-out from training)
  - Cameras: robot0_agentview_left @ 128x128 (primary), + right + wrist available
  - Reports: per-task success rate + per-split average + overall average

Usage (zero-shot with pretrained weights):
    python scripts/eval_robocasa365.py \
        --model_path CogACT/CogACT-Base \
        --norm_stats_path data/robocasa/dataset_statistics.json \
        --action_model_type DiT-B \
        --trials_per_scene 10 \
        --output_dir results/eval_robocasa365

Usage (finetuned model):
    python scripts/eval_robocasa365.py \
        --model_path runs/robocasa/cogact-robocasa-CogACT-Base-freeze=dit_only \
        --unnorm_key robocasa \
        --trials_per_scene 10

Task splits (from robocasa/utils/dataset_registry.py):
  atomic_seen (18):
    CloseBlenderLid, CloseFridge, CloseToasterOvenDoor, CoffeeSetupMug, NavigateKitchen,
    OpenCabinet, OpenDrawer, OpenStandMixerHead, PickPlaceCounterToCabinet,
    PickPlaceCounterToStove, PickPlaceDrawerToCounter, PickPlaceSinkToCounter,
    PickPlaceToasterToCounter, SlideDishwasherRack, TurnOffStove, TurnOnElectricKettle,
    TurnOnMicrowave, TurnOnSinkFaucet

  composite_seen (16):
    DeliverStraw, GetToastedBread, KettleBoiling, LoadDishwasher, PackIdenticalLunches,
    PreSoakPan, PrepareCoffee, RinseSinkBasin, ScrubCuttingBoard, SearingMeat,
    SetUpCuttingStation, StackBowlsCabinet, SteamInMicrowave, StirVegetables,
    StoreLeftoversInBowl, WashLettuce

  composite_unseen (16):
    ArrangeBreadBasket, ArrangeTea, BreadSelection, CategorizeCondiments,
    CuttingToolSelection, GarnishPancake, GatherTableware, HeatKebabSandwich,
    MakeIceLemonade, PanTransfer, PortionHotDogs, RecycleBottlesByType,
    SeparateFreezerRack, WaffleReheat, WashFruitColander, WeighIngredients
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────────────────────────────────────
# RoboCasa365 exact task split definitions
# (from robocasa/utils/dataset_registry.py, v1.0 release)
# ──────────────────────────────────────────────────────────────────────────────

ATOMIC_SEEN_TASKS: List[str] = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]

COMPOSITE_SEEN_TASKS: List[str] = [
    "DeliverStraw",
    "GetToastedBread",
    "KettleBoiling",
    "LoadDishwasher",
    "PackIdenticalLunches",
    "PreSoakPan",
    "PrepareCoffee",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SetUpCuttingStation",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WashLettuce",
]

COMPOSITE_UNSEEN_TASKS: List[str] = [
    "ArrangeBreadBasket",
    "ArrangeTea",
    "BreadSelection",
    "CategorizeCondiments",
    "CuttingToolSelection",
    "GarnishPancake",
    "GatherTableware",
    "HeatKebabSandwich",
    "MakeIceLemonade",
    "PanTransfer",
    "PortionHotDogs",
    "RecycleBottlesByType",
    "SeparateFreezerRack",
    "WaffleReheat",
    "WashFruitColander",
    "WeighIngredients",
]

TASK_SPLITS: Dict[str, List[str]] = {
    "atomic_seen": ATOMIC_SEEN_TASKS,
    "composite_seen": COMPOSITE_SEEN_TASKS,
    "composite_unseen": COMPOSITE_UNSEEN_TASKS,
}

# 5 fixed evaluation scenes: (layout_id, style_id)
# Matches robocasa/utils/eval_utils.py layout config
EVAL_SCENES: List[Tuple[int, int]] = [
    (1, 1),
    (2, 2),
    (4, 4),
    (6, 9),
    (7, 10),
]

# Default episode horizons (steps)
ATOMIC_HORIZON = 400
COMPOSITE_HORIZON = 600


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RoboCasa365 evaluation protocol for CogACT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model_path", type=str, required=True,
                   help="HF Hub repo ID (e.g. 'CogACT/CogACT-Base') or local run directory "
                        "or direct path to a .pt checkpoint.")
    p.add_argument("--action_model_type", type=str, default="DiT-B",
                   choices=["DiT-S", "DiT-B", "DiT-L"])
    p.add_argument("--future_action_window_size", type=int, default=15)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--unnorm_key", type=str, default=None,
                   help="Key used for action un-normalisation. Pass 'robocasa' for a "
                        "finetuned model, or a pretraining dataset key for zero-shot eval. "
                        "If None, the script will try to infer from the model's norm_stats.")
    p.add_argument("--norm_stats_path", type=str, default=None,
                   help="Path to a dataset_statistics.json to patch into the model for "
                        "zero-shot evaluation on RoboCasa. Required if the pretrained model "
                        "has no 'robocasa' key in its norm_stats.")

    # Eval scope
    p.add_argument("--task_set", type=str, nargs="+",
                   default=["atomic_seen", "composite_seen", "composite_unseen"],
                   choices=list(TASK_SPLITS.keys()),
                   help="Which splits to evaluate.")
    p.add_argument("--tasks", type=str, nargs="*", default=None,
                   help="Override: evaluate only these specific task names.")
    p.add_argument("--trials_per_scene", type=int, default=10,
                   help="Episodes per (layout, style) scene. "
                        "Total trials per task = trials_per_scene × 5 scenes.")

    # Environment
    p.add_argument("--robot", type=str, default="Panda",
                   help="Robot type. Use 'Panda' (7-DoF, matches CogACT) or 'PandaMobile'.")
    p.add_argument("--controller", type=str, default="OSC_POSE")
    p.add_argument("--camera_name", type=str, default="robot0_agentview_left",
                   help="Primary camera fed to CogACT.")
    p.add_argument("--img_size", type=int, default=128,
                   help="Camera resolution (128 is the RoboCasa365 standard).")
    p.add_argument("--object_instance_split", type=str, default="B",
                   help="Object instance split. 'B' = held-out eval instances.")
    p.add_argument("--atomic_horizon", type=int, default=ATOMIC_HORIZON)
    p.add_argument("--composite_horizon", type=int, default=COMPOSITE_HORIZON)

    # Inference
    p.add_argument("--cfg_scale", type=float, default=1.5)
    p.add_argument("--use_ddim", action="store_true", default=True)
    p.add_argument("--num_ddim_steps", type=int, default=10)
    p.add_argument("--action_exec_horizon", type=int, default=1,
                   help="How many actions from each predicted chunk to execute before "
                        "re-predicting. 1 = re-predict every step (slowest, most reactive). "
                        "Use future_action_window_size+1 to execute the full chunk.")

    # Output / reproducibility
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def resolve_checkpoint(model_path: str) -> str:
    p = Path(model_path)
    if p.is_file() and p.suffix == ".pt":
        return str(p)
    if p.is_dir():
        ckpts = sorted((p / "checkpoints").glob("*.pt"))
        assert ckpts, f"No .pt checkpoint found in {p / 'checkpoints'}"
        return str(ckpts[-1])
    return model_path  # HF Hub repo ID


def load_model(args: argparse.Namespace):
    from vla import load_vla

    token = args.hf_token or os.environ.get("HF_TOKEN")
    checkpoint = resolve_checkpoint(args.model_path)

    print(f"Loading CogACT from: {checkpoint}")
    model = load_vla(
        checkpoint,
        hf_token=token,
        load_for_training=False,
        action_model_type=args.action_model_type,
        action_dim=args.action_dim,
        future_action_window_size=args.future_action_window_size,
        past_action_window_size=0,
        use_ema=False,
    )

    # Patch in external norm_stats if provided
    if args.norm_stats_path:
        with open(args.norm_stats_path) as f:
            extra_stats = json.load(f)
        if model.norm_stats is None:
            model.norm_stats = {}
        model.norm_stats.update(extra_stats)
        print(f"Patched norm_stats from {args.norm_stats_path}: keys = {list(extra_stats.keys())}")

    # Resolve unnorm_key
    if args.unnorm_key is None:
        if model.norm_stats and "robocasa" in model.norm_stats:
            args.unnorm_key = "robocasa"
        elif model.norm_stats and len(model.norm_stats) == 1:
            args.unnorm_key = next(iter(model.norm_stats))
            print(f"Auto-selected unnorm_key='{args.unnorm_key}'")
        else:
            available = list(model.norm_stats.keys()) if model.norm_stats else []
            raise ValueError(
                f"Cannot infer unnorm_key. Model has keys: {available}. "
                "Pass --unnorm_key or --norm_stats_path."
            )

    model = model.to(args.device).eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_env(
    task_name: str,
    layout_id: int,
    style_id: int,
    robot: str,
    controller: str,
    camera_name: str,
    img_size: int,
    object_instance_split: str,
    seed: Optional[int] = None,
):
    """Create a single RoboCasa environment for the given task + scene config."""
    import robosuite as suite
    try:
        import robocasa.environments  # noqa: F401 — registers all RoboCasa envs
    except ImportError:
        import robocasa  # noqa: F401

    env_kwargs = dict(
        env_name=task_name,
        robots=robot,
        controller_configs=suite.load_controller_config(default_controller=controller),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_object_obs=False,
        use_camera_obs=True,
        camera_names=[camera_name],
        camera_heights=img_size,
        camera_widths=img_size,
        layout_ids=layout_id,
        style_ids=style_id,
        obj_instance_split=object_instance_split,
        translucent_robot=False,
    )
    if seed is not None:
        env_kwargs["seed"] = seed

    return suite.make(**env_kwargs)


def get_instruction(env) -> str:
    """Extract language instruction from the RoboCasa environment."""
    try:
        ep_meta = env.get_ep_meta()
        if isinstance(ep_meta, dict) and "lang" in ep_meta:
            return ep_meta["lang"]
    except (AttributeError, KeyError):
        pass
    # Fall back to env name
    env_name = type(env).__name__
    return env_name.replace("_", " ").lower()


# ──────────────────────────────────────────────────────────────────────────────
# Single-episode rollout
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(
    model,
    env,
    instruction: str,
    horizon: int,
    camera_name: str,
    unnorm_key: str,
    cfg_scale: float,
    use_ddim: bool,
    num_ddim_steps: int,
    action_exec_horizon: int,
    device: str,
) -> bool:
    """Run one episode; return True if successful."""
    obs = env.reset()
    success = False

    action_chunk = None
    chunk_idx = 0  # position within the current action chunk

    for step in range(horizon):
        # Re-predict when chunk is exhausted or on first step
        if action_chunk is None or chunk_idx >= action_exec_horizon:
            img_np = obs[f"{camera_name}_image"]
            if img_np.ndim == 4:
                img_np = img_np[0]
            pil_img = Image.fromarray(img_np.astype(np.uint8))

            with torch.no_grad():
                action_chunk, _ = model.predict_action(
                    image=pil_img,
                    instruction=instruction,
                    unnorm_key=unnorm_key,
                    cfg_scale=cfg_scale,
                    use_ddim=use_ddim,
                    num_ddim_steps=num_ddim_steps,
                    do_sample=False,
                )
            chunk_idx = 0

        action = action_chunk[chunk_idx]  # [action_dim]
        chunk_idx += 1

        obs, _reward, done, info = env.step(action)

        if info.get("success", False):
            success = True
            break
        if done:
            break

    return success


# ──────────────────────────────────────────────────────────────────────────────
# Per-task evaluation over all 5 scenes
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_task(
    model,
    task_name: str,
    is_composite: bool,
    args: argparse.Namespace,
    scene_seed_offset: int = 0,
) -> Dict:
    """
    Evaluate one task across all 5 fixed evaluation scenes.
    Returns a dict with per-scene and aggregate results.
    """
    horizon = args.composite_horizon if is_composite else args.atomic_horizon
    all_successes = []
    scene_results = []

    for scene_idx, (layout_id, style_id) in enumerate(EVAL_SCENES):
        scene_successes = []
        for trial in range(args.trials_per_scene):
            trial_seed = args.seed + scene_seed_offset + scene_idx * 1000 + trial
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
                    seed=trial_seed,
                )
                instruction = get_instruction(env)
                success = run_episode(
                    model=model,
                    env=env,
                    instruction=instruction,
                    horizon=horizon,
                    camera_name=args.camera_name,
                    unnorm_key=args.unnorm_key,
                    cfg_scale=args.cfg_scale,
                    use_ddim=args.use_ddim,
                    num_ddim_steps=args.num_ddim_steps,
                    action_exec_horizon=args.action_exec_horizon,
                    device=args.device,
                )
                env.close()
            except Exception as e:
                print(f"  [WARN] {task_name} scene={scene_idx} trial={trial} failed: {e}")
                success = False

            scene_successes.append(float(success))
            all_successes.append(float(success))

            if args.verbose:
                print(
                    f"  {task_name} | scene ({layout_id},{style_id}) "
                    f"trial {trial+1}/{args.trials_per_scene} | "
                    f"success={success}"
                )

        scene_results.append({
            "layout_id": layout_id,
            "style_id": style_id,
            "success_rate": float(np.mean(scene_successes)),
            "n_trials": len(scene_successes),
        })

    return {
        "task": task_name,
        "success_rate": float(np.mean(all_successes)),
        "n_trials": len(all_successes),
        "scene_results": scene_results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    # Verify robosuite / robocasa are importable
    try:
        import robosuite  # noqa: F401
    except ImportError:
        raise ImportError(
            "robosuite not found. Run: bash scripts/setup_robocasa.sh"
        )

    # Build task list
    if args.tasks:
        # Manual override: evaluate specific tasks only
        tasks_to_eval = [(t, _infer_split(t)) for t in args.tasks]
    else:
        tasks_to_eval = []
        for split_name in args.task_set:
            for t in TASK_SPLITS[split_name]:
                tasks_to_eval.append((t, split_name))

    total_tasks = len(tasks_to_eval)
    total_trials = total_tasks * len(EVAL_SCENES) * args.trials_per_scene
    print(f"\n=== RoboCasa365 Evaluation ===")
    print(f"  Model:          {args.model_path}")
    print(f"  Splits:         {args.task_set}")
    print(f"  Tasks:          {total_tasks}")
    print(f"  Scenes/task:    {len(EVAL_SCENES)} (fixed)")
    print(f"  Trials/scene:   {args.trials_per_scene}")
    print(f"  Total trials:   {total_trials}")
    print(f"  Robot:          {args.robot}")
    print(f"  Camera:         {args.camera_name} @ {args.img_size}px")
    print(f"  unnorm_key:     {args.unnorm_key or '(to be resolved after model load)'}")
    print()

    model = load_model(args)
    print(f"  unnorm_key:     {args.unnorm_key}")
    print()

    # ── Run evaluation ──
    all_task_results = []
    split_results: Dict[str, List[float]] = {s: [] for s in TASK_SPLITS}

    t0 = time.time()
    for task_idx, (task_name, split_name) in enumerate(
        tqdm(tasks_to_eval, desc="Tasks", unit="task")
    ):
        is_composite = split_name in ("composite_seen", "composite_unseen")
        print(f"\n[{task_idx+1}/{total_tasks}] {task_name} ({split_name})")

        task_result = evaluate_task(
            model=model,
            task_name=task_name,
            is_composite=is_composite,
            args=args,
            scene_seed_offset=task_idx * 10000,
        )
        task_result["split"] = split_name
        all_task_results.append(task_result)

        sr = task_result["success_rate"]
        split_results[split_name].append(sr)
        print(f"  → success_rate = {sr:.1%}  ({task_result['n_trials']} trials)")

    elapsed = time.time() - t0

    # ── Aggregate results ──
    split_averages = {
        split: float(np.mean(rates)) if rates else float("nan")
        for split, rates in split_results.items()
        if rates  # only include evaluated splits
    }
    overall_sr = float(np.mean([r["success_rate"] for r in all_task_results]))

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for split, avg in split_averages.items():
        n = len(split_results[split])
        print(f"  {split:<22} {avg:.1%}  (avg over {n} tasks)")
    print(f"  {'OVERALL':<22} {overall_sr:.1%}  ({total_tasks} tasks)")
    print(f"  Elapsed: {elapsed/60:.1f} min")

    # ── Save results ──
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif Path(args.model_path).is_dir():
        out_dir = Path(args.model_path) / "eval_robocasa365"
    else:
        out_dir = Path("results") / "eval_robocasa365"
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "model_path": args.model_path,
        "unnorm_key": args.unnorm_key,
        "robot": args.robot,
        "camera": args.camera_name,
        "img_size": args.img_size,
        "trials_per_scene": args.trials_per_scene,
        "total_trials": total_trials,
        "elapsed_seconds": elapsed,
        "split_averages": split_averages,
        "overall_success_rate": overall_sr,
        "per_task": all_task_results,
    }

    out_file = out_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {out_file}")

    # Also write a compact summary CSV for easy viewing
    csv_file = out_dir / "summary.csv"
    with open(csv_file, "w") as f:
        f.write("task,split,success_rate,n_trials\n")
        for r in all_task_results:
            f.write(f"{r['task']},{r['split']},{r['success_rate']:.4f},{r['n_trials']}\n")
        f.write(f"\n")
        for split, avg in split_averages.items():
            f.write(f"AVG_{split.upper()},,-,{avg:.4f},-\n")
        f.write(f"OVERALL,,{overall_sr:.4f},{total_trials}\n")
    print(f"Summary CSV saved to:  {csv_file}")


def _infer_split(task_name: str) -> str:
    for split, tasks in TASK_SPLITS.items():
        if task_name in tasks:
            return split
    return "unknown"


if __name__ == "__main__":
    main()
