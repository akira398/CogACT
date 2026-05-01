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
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    p.add_argument("--object_instance_split", type=str, default="target",
                   help="Object instance split. 'target' = held-out eval instances (robocasa v1.0). "
                        "Use 'pretrain' for training objects, None for all.")
    p.add_argument("--num_scenes", type=int, default=None,
                   help="Limit evaluation to the first N of the 5 fixed scenes per task. "
                        "Use 1 for a quick sanity check (1 scene × trials_per_scene trials).")
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

    # Video recording
    p.add_argument("--record_n_videos", type=int, default=0,
                   help="Record N randomly chosen trials as MP4 (re-runs after eval). "
                        "Videos saved to <output_dir>/videos/.")
    p.add_argument("--video_fps", type=int, default=10)
    p.add_argument("--video_size", type=int, default=256,
                   help="Camera resolution for recorded videos (independent of --img_size "
                        "used for model inference). Higher = better quality.")
    p.add_argument("--gt_data_root", type=str, default=None,
                   help="Path to downloaded demo data (Parquet or HDF5) for GT replay videos. "
                        "E.g. datasets/robocasa/v1.0/target")
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

def _load_controller_config(controller_name: str) -> dict:
    """Load robosuite controller config; handles API rename across versions."""
    import robosuite as suite
    loader = getattr(suite, "load_controller_config",
                     getattr(suite, "load_part_controller_config", None))
    if loader is None:
        raise ImportError(
            "robosuite has neither load_controller_config nor load_part_controller_config"
        )
    return loader(default_controller=controller_name)


def _patch_robosuite_compat() -> None:
    """Make robosuite 1.5.x classes silently drop unknown kwargs from robocasa v1.0.

    robocasa v1.0 was written against a robosuite dev version that added several
    new __init__ parameters (load_model_on_init, enable_multiccd,
    enable_sleeping_islands, ...). None exist in the released 1.5.x. Rather than
    patching them one by one, we wrap the two affected classes to filter out any
    kwarg that their original signature doesn't accept.
    """
    import inspect

    def _make_permissive(cls) -> None:
        sig = inspect.signature(cls.__init__)
        # Already accepts **kwargs — nothing to do
        if any(p.kind == inspect.Parameter.VAR_KEYWORD
               for p in sig.parameters.values()):
            return
        valid = frozenset(sig.parameters.keys()) - {"self"}
        _orig = cls.__init__
        def _patched(self, *args, **kwargs):
            dropped = {k for k in kwargs if k not in valid}
            if dropped:
                for k in dropped:
                    kwargs.pop(k)
            return _orig(self, *args, **kwargs)
        cls.__init__ = _patched

    try:
        from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
        _make_permissive(ManipulationEnv)
    except Exception:
        pass

    try:
        from robosuite.models.tasks import Task
        _make_permissive(Task)
    except Exception:
        pass

    # robosuite 1.5.2 calls initialize_renderer() during __init__ before the
    # full kitchen scene (with agentview cameras) is assembled.  Catch the
    # camera-not-found ValueError so the offscreen renderer can still be set up.
    try:
        from robosuite.environments.base import MujocoEnv
        _orig_renderer = MujocoEnv.initialize_renderer
        def _safe_renderer(self):
            try:
                _orig_renderer(self)
            except ValueError as e:
                if "camera" in str(e).lower():
                    pass  # kitchen cameras not assembled yet — safe to ignore
                else:
                    raise
        MujocoEnv.initialize_renderer = _safe_renderer
    except Exception:
        pass

    # Observable._check_sensor_validity fires during __init__ and tries to
    # render from each camera to determine its output shape.  The kitchen
    # agentview cameras only exist after the first _reset_internal, so the
    # render fails.  Suppress the error; _data_shape will be resolved on first
    # actual use after env.reset() assembles the full kitchen model.
    try:
        from robosuite.utils.observables import Observable
        _orig_check_sensor = Observable._check_sensor_validity
        def _lenient_check_sensor(self):
            try:
                _orig_check_sensor(self)
            except ValueError:
                self._data_shape = None
        Observable._check_sensor_validity = _lenient_check_sensor
    except Exception:
        pass

    # robocasa's robot-placement functions hardcode mobile-base joint names that
    # only exist on PandaMobile.  For a fixed-base Panda, skip all base-motion
    # ops: set_robot_base returns anchor_pos (stored as init_robot_base_pos);
    # set_robot_to_position becomes a no-op (robot is already in its XML pose).
    try:
        import numpy as _np
        import robocasa.utils.env_utils as _eu

        def _has_mobile_base(env):
            try:
                env.sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_yaw")
                return True
            except (ValueError, AttributeError):
                return False

        _orig_set_robot_base = _eu.set_robot_base
        _orig_set_robot_to_position = _eu.set_robot_to_position

        def _patched_set_robot_base(env, anchor_pos, anchor_ori, rot_dev, pos_dev_x, pos_dev_y):
            if not _has_mobile_base(env):
                return _np.asarray(anchor_pos, dtype=float)
            return _orig_set_robot_base(env, anchor_pos, anchor_ori, rot_dev, pos_dev_x, pos_dev_y)

        def _patched_set_robot_to_position(env, global_pos):
            if not _has_mobile_base(env):
                return
            return _orig_set_robot_to_position(env, global_pos)

        _eu.set_robot_base = _patched_set_robot_base
        _eu.set_robot_to_position = _patched_set_robot_to_position
    except Exception:
        pass

    # robocasa's DEFAULT camera configs attach agentview cameras to
    # 'mobilebase0_support' (PandaMobile only).  For fixed-base Panda, that
    # body doesn't exist so edit_model_xml silently skips the cameras.
    # Add a "Panda" entry that remaps them to 'robot0_base' instead.
    try:
        from copy import deepcopy
        from robocasa.utils import camera_utils as _cu

        if "Panda" not in _cu.CAM_CONFIGS:
            panda_overrides = {}
            for _cam_name, _cam_cfg in _cu.CAM_CONFIGS.get("DEFAULT", {}).items():
                if _cam_cfg.get("parent_body") == "mobilebase0_support":
                    _cfg = deepcopy(_cam_cfg)
                    _cfg["parent_body"] = "robot0_base"
                    panda_overrides[_cam_name] = _cfg
            if panda_overrides:
                _cu.CAM_CONFIGS["Panda"] = panda_overrides
    except Exception:
        pass

    # robosuite 1.5.x defaults to IMAGE_CONVENTION="opengl" which skips the
    # vertical flip (img[::1] = no-op).  Raw MuJoCo frames are bottom-to-top,
    # so without the flip images come out upside-down — both for the model
    # input and for recorded videos.  Force "opencv" so img[::-1] is applied.
    try:
        import robosuite.macros as _macros
        _macros.IMAGE_CONVENTION = "opencv"
    except Exception:
        pass


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
    _patch_robosuite_compat()

    env_kwargs = dict(
        env_name=task_name,
        robots=robot,
        controller_configs=_load_controller_config(controller),
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
    record: bool = False,
    inference_size: Optional[int] = None,
) -> Tuple[bool, Optional[List[np.ndarray]]]:
    """Run one episode; return (success, frames). frames is None unless record=True.

    inference_size: resize image to this square size before model inference.
    Useful when the env is created at higher resolution for video recording.
    """
    obs = env.reset()
    success = False
    frames: Optional[List[np.ndarray]] = [] if record else None

    action_chunk = None
    chunk_idx = 0

    for step in range(horizon):
        # Grab current frame (always, to avoid duplicate obs fetches)
        img_np = obs[f"{camera_name}_image"]
        if img_np.ndim == 4:
            img_np = img_np[0]

        if frames is not None:
            frames.append(img_np.copy())

        # Re-predict when chunk is exhausted or on first step
        if action_chunk is None or chunk_idx >= action_exec_horizon:
            pil_img = Image.fromarray(img_np.astype(np.uint8))
            if inference_size is not None and (img_np.shape[0] != inference_size or img_np.shape[1] != inference_size):
                pil_img = pil_img.resize((inference_size, inference_size), Image.BILINEAR)
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

    return success, frames


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

    scenes = EVAL_SCENES[:args.num_scenes] if args.num_scenes else EVAL_SCENES
    for scene_idx, (layout_id, style_id) in enumerate(scenes):
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
                success, _ = run_episode(
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
# Video recording helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_video(frames: List[np.ndarray], path: Path, fps: int = 10) -> None:
    """Save a list of HWC uint8 RGB frames as MP4."""
    if not frames:
        return
    try:
        import imageio
        imageio.mimsave(str(path), [f.astype(np.uint8) for f in frames], fps=fps)
    except ImportError:
        import cv2
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        for frame in frames:
            out.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
        out.release()


def _load_gt_actions(task_name: str, gt_data_root: Path) -> Optional[np.ndarray]:
    """Return a (T, action_dim) array for a random GT demo of task_name, or None."""
    # ── Parquet (RoboCasa365 v1.0) ──
    parquet_files = [f for f in gt_data_root.rglob("*.parquet")
                     if task_name in str(f)]
    if parquet_files:
        try:
            import pandas as pd
            pq_file = random.choice(parquet_files)
            df = pd.read_parquet(pq_file)
            action_col = next((c for c in ("action", "actions") if c in df.columns), None)
            if action_col is None:
                return None
            if "episode_index" in df.columns:
                ep_idx = int(df["episode_index"].sample(1).iloc[0])
                df = df[df["episode_index"] == ep_idx]
            return np.stack(df[action_col].to_list()).astype(np.float32)
        except Exception as e:
            print(f"  [GT] Parquet load error for {task_name}: {e}")

    # ── HDF5 (RoboCasa v0.2) ──
    hdf5_files = [f for f in gt_data_root.rglob("*.hdf5")
                  if task_name in str(f)]
    if hdf5_files:
        try:
            import h5py
            hdf5_file = random.choice(hdf5_files)
            with h5py.File(str(hdf5_file), "r") as f:
                if "data" in f:
                    demo_key = random.choice(list(f["data"].keys()))
                    return f["data"][demo_key]["actions"][:].astype(np.float32)
        except Exception as e:
            print(f"  [GT] HDF5 load error for {task_name}: {e}")

    return None


def _record_gt_video(
    task_name: str,
    gt_data_root: Path,
    env_kwargs: dict,
    camera_name: str,
    out_path: Path,
    fps: int,
) -> None:
    """Save a GT demo video.

    First tries to copy a pre-rendered MP4 directly from the LeRobot dataset
    (fast, no simulation needed, correct camera).  Falls back to simulation
    replay only if no pre-rendered video is found.
    """
    import shutil

    # ── Try direct MP4 copy from LeRobot dataset ──────────────────────────────
    # Dataset structure: <gt_data_root>/<task>/videos/chunk-000/<cam_key>/episode_XXXXXX.mp4
    # camera_name = "robot0_agentview_left"
    # cam_key     = "observation.images.robot0_agentview_left"  (LeRobot convention)
    task_ds = gt_data_root / task_name
    if task_ds.exists():
        vid_root = task_ds / "videos" / "chunk-000"
        # find the right subdirectory: exact key first, then suffix match
        cam_dir = None
        exact = vid_root / f"observation.images.{camera_name}"
        if exact.is_dir():
            cam_dir = exact
        else:
            suffix = camera_name.split("robot0_")[-1]  # e.g. "agentview_left"
            for d in sorted(vid_root.iterdir()) if vid_root.is_dir() else []:
                if suffix in d.name:
                    cam_dir = d
                    break
        if cam_dir is not None:
            mp4s = sorted(cam_dir.glob("episode_*.mp4"))
            if mp4s:
                shutil.copy(mp4s[0], out_path)
                print(f"    GT video:     {out_path}  (copied from dataset)")
                return

    # ── Fallback: replay GT actions in simulation ─────────────────────────────
    actions = _load_gt_actions(task_name, gt_data_root)
    if actions is None:
        print(f"  [GT] No demo data found for {task_name} — skipping GT video.")
        return

    try:
        import robosuite as suite
        env = suite.make(**env_kwargs)
        obs = env.reset()
        frames = []
        for action in actions:
            img_np = obs[f"{camera_name}_image"]
            if img_np.ndim == 4:
                img_np = img_np[0]
            frames.append(img_np.copy())
            obs, _, done, _ = env.step(action[:env.action_spec[0].shape[0]])
            if done:
                break
        env.close()
        save_video(frames, out_path, fps)
        print(f"    GT video:     {out_path}  (simulation replay)")
    except Exception as e:
        print(f"  [GT] Failed to record GT video for {task_name}: {e}")


def record_videos(
    model: Any,
    tasks_to_eval: List[Tuple[str, str]],
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    """Re-run N randomly selected trials with video recording after the main eval."""
    video_dir = out_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    # Build a flat list of every (task, scene, trial) spec
    all_specs = []
    for task_idx, (task_name, split_name) in enumerate(tasks_to_eval):
        is_composite = split_name in ("composite_seen", "composite_unseen")
        scenes = EVAL_SCENES[:args.num_scenes] if args.num_scenes else EVAL_SCENES
        for scene_idx, (layout_id, style_id) in enumerate(scenes):
            for trial in range(args.trials_per_scene):
                trial_seed = args.seed + task_idx * 10000 + scene_idx * 1000 + trial
                all_specs.append({
                    "task_name": task_name,
                    "split": split_name,
                    "is_composite": is_composite,
                    "layout_id": layout_id,
                    "style_id": style_id,
                    "trial_seed": trial_seed,
                    "label": f"{task_name}_scene{scene_idx}_trial{trial}",
                })

    chosen = random.sample(all_specs, min(args.record_n_videos, len(all_specs)))
    print(f"\nRecording {len(chosen)} video(s) → {video_dir}")

    gt_data_root = Path(args.gt_data_root) if args.gt_data_root else None

    for spec in chosen:
        label = spec["label"]
        horizon = args.composite_horizon if spec["is_composite"] else args.atomic_horizon
        env_kwargs = dict(
            env_name=spec["task_name"],
            robots=args.robot,
            controller_configs=_load_controller_config(args.controller),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_object_obs=False,
            use_camera_obs=True,
            camera_names=[args.camera_name],
            camera_heights=args.video_size,
            camera_widths=args.video_size,
            layout_ids=spec["layout_id"],
            style_ids=spec["style_id"],
            obj_instance_split=args.object_instance_split,
            translucent_robot=False,
            seed=spec["trial_seed"],
        )

        print(f"\n  [{label}]")
        try:
            import robosuite as suite
            try:
                import robocasa.environments  # noqa: F401
            except ImportError:
                import robocasa  # noqa: F401
            env = suite.make(**env_kwargs)
            instruction = get_instruction(env)
            success, frames = run_episode(
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
                record=True,
                inference_size=args.img_size,
            )
            env.close()

            suffix = "success" if success else "fail"
            policy_path = video_dir / f"{label}_policy_{suffix}.mp4"
            save_video(frames, policy_path, fps=args.video_fps)
            print(f"    Policy video: {policy_path}  (success={success})")
        except Exception as e:
            print(f"    [WARN] Policy video failed: {e}")

        if gt_data_root:
            gt_path = video_dir / f"{label}_GT.mp4"
            _record_gt_video(
                task_name=spec["task_name"],
                gt_data_root=gt_data_root,
                env_kwargs=env_kwargs,
                camera_name=args.camera_name,
                out_path=gt_path,
                fps=args.video_fps,
            )


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
    n_scenes = min(args.num_scenes, len(EVAL_SCENES)) if args.num_scenes else len(EVAL_SCENES)
    total_trials = total_tasks * n_scenes * args.trials_per_scene
    print(f"\n=== RoboCasa365 Evaluation ===")
    print(f"  Model:          {args.model_path}")
    print(f"  Splits:         {args.task_set}")
    print(f"  Tasks:          {total_tasks}")
    print(f"  Scenes/task:    {n_scenes}{' (limited)' if args.num_scenes else ' (all 5)'}")
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

    # ── Record sample videos ──
    if args.record_n_videos > 0:
        record_videos(model, tasks_to_eval, args, out_dir)


def _infer_split(task_name: str) -> str:
    for split, tasks in TASK_SPLITS.items():
        if task_name in tasks:
            return split
    return "unknown"


if __name__ == "__main__":
    main()
