"""
eval_robocasa.py

Evaluate a CogACT model (pretrained or finetuned) on RoboCasa simulation tasks.

Supports two evaluation modes:
  --mode online   Run live rollouts in the RoboCasa simulator and report task success rate.
  --mode offline  Compute action prediction MSE on held-out HDF5 demonstrations (no simulator).

Requirements (online mode):
    pip install robosuite
    pip install -e third_party/robocasa   (see scripts/setup_robocasa.sh)

Usage (online):
    python scripts/eval_robocasa.py \
        --model_path runs/robocasa/cogact-robocasa-CogACT-Base-freeze=dit_only \
        --task PnPCounterToCab \
        --num_episodes 50 \
        --mode online

Usage (offline):
    python scripts/eval_robocasa.py \
        --model_path runs/robocasa/cogact-robocasa-CogACT-Base-freeze=dit_only \
        --data_root data/robocasa/PnPCounterToCab \
        --mode offline
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CogACT on RoboCasa.")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to the finetuned run directory OR a HF Hub repo ID.")
    p.add_argument("--task", type=str, default="PnPCounterToCab",
                   help="RoboCasa task name (online mode).")
    p.add_argument("--data_root", type=str, default=None,
                   help="Path to .hdf5 data for offline mode.")
    p.add_argument("--mode", type=str, default="online",
                   choices=["online", "offline"],
                   help="Evaluation mode.")
    p.add_argument("--num_episodes", type=int, default=50,
                   help="Number of episodes / demos to evaluate.")
    p.add_argument("--max_steps", type=int, default=400,
                   help="Max environment steps per episode (online).")
    p.add_argument("--camera_name", type=str, default="robot0_agentview_left",
                   help="Camera to feed into the model.")
    p.add_argument("--image_size", type=int, default=224,
                   help="Image resolution fed to the model.")
    p.add_argument("--action_model_type", type=str, default="DiT-B",
                   choices=["DiT-S", "DiT-B", "DiT-L"],
                   help="Action model variant (must match the checkpoint).")
    p.add_argument("--future_action_window_size", type=int, default=15)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--cfg_scale", type=float, default=1.5,
                   help="Classifier-free guidance scale (1.0 = disabled).")
    p.add_argument("--use_ddim", action="store_true", default=True)
    p.add_argument("--num_ddim_steps", type=int, default=10)
    p.add_argument("--unnorm_key", type=str, default="robocasa",
                   help="Key for action un-normalisation statistics.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory to save results JSON. Defaults to <model_path>/eval.")
    p.add_argument("--hf_token", type=str, default=None)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_checkpoint(model_path: str) -> str:
    """
    Accept either:
      1. A run directory (contains config.json + checkpoints/*.pt)
      2. A direct path to a .pt file
      3. A HuggingFace Hub repo ID
    Returns the value suitable for load_vla().
    """
    p = Path(model_path)
    if p.is_file() and p.suffix == ".pt":
        return str(p)
    if p.is_dir():
        ckpts = sorted((p / "checkpoints").glob("*.pt"))
        assert ckpts, f"No .pt checkpoints found in {p / 'checkpoints'}"
        return str(ckpts[-1])  # latest by alphabetical sort
    # Assume HF Hub repo
    return model_path


def _resolve_dataset_stats(model_path: str, unnorm_key: str) -> Optional[dict]:
    """
    Try to load dataset_statistics.json from the run directory.
    Returns None if not found (model will use its own built-in stats).
    """
    p = Path(model_path)
    if p.is_dir():
        stats_file = p / "dataset_statistics.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            if unnorm_key in stats:
                return stats
    return None


def load_model(args: argparse.Namespace):
    """Load CogACT model and patch norm_stats if a finetuned run dir is given."""
    from vla import load_vla

    token = args.hf_token or os.environ.get("HF_TOKEN")
    checkpoint = _resolve_checkpoint(args.model_path)

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

    # Patch in RoboCasa normalization stats if available
    extra_stats = _resolve_dataset_stats(args.model_path, args.unnorm_key)
    if extra_stats is not None and args.unnorm_key not in (model.norm_stats or {}):
        if model.norm_stats is None:
            model.norm_stats = {}
        model.norm_stats.update(extra_stats)
        print(f"Patched norm_stats with key '{args.unnorm_key}' from run directory.")

    model = model.to(args.device).eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Online evaluation in the RoboCasa simulator
# ──────────────────────────────────────────────────────────────────────────────

def eval_online(args: argparse.Namespace, model) -> dict:
    try:
        import robosuite as suite
        import robocasa  # noqa: F401 — registers RoboCasa environments
        import robocasa.environments  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Online evaluation requires robosuite and robocasa. "
            "Run: bash scripts/setup_robocasa.sh"
        ) from e

    np.random.seed(args.seed)
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")

    env = suite.make(
        env_name=args.task,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_object_obs=False,
        use_camera_obs=True,
        camera_names=[args.camera_name],
        camera_heights=args.image_size,
        camera_widths=args.image_size,
        reward_shaping=False,
        horizon=args.max_steps,
    )

    successes = []
    episode_lengths = []

    for ep_idx in tqdm(range(args.num_episodes), desc=f"Eval {args.task}"):
        obs = env.reset()

        # Retrieve language instruction from the environment's episode metadata
        try:
            ep_meta = env.get_ep_meta()
            instruction = ep_meta.get("lang", args.task.replace("_", " ").lower())
        except AttributeError:
            instruction = args.task.replace("_", " ").lower()

        success = False
        for step_idx in range(args.max_steps):
            # Extract camera image: RoboCasa returns [H, W, 3] uint8
            img_np = obs[f"{args.camera_name}_image"]
            if img_np.ndim == 4:
                img_np = img_np[0]  # remove batch dim if present

            pil_image = Image.fromarray(img_np.astype(np.uint8))

            with torch.no_grad():
                actions, _ = model.predict_action(
                    image=pil_image,
                    instruction=instruction,
                    unnorm_key=args.unnorm_key,
                    cfg_scale=args.cfg_scale,
                    use_ddim=args.use_ddim,
                    num_ddim_steps=args.num_ddim_steps,
                    do_sample=False,
                )

            # Execute the first predicted action in the chunk
            action = actions[0]  # [action_dim]
            obs, reward, done, info = env.step(action)

            if info.get("success", False) or done:
                success = info.get("success", False)
                episode_lengths.append(step_idx + 1)
                break
        else:
            episode_lengths.append(args.max_steps)

        successes.append(float(success))

    env.close()

    results = {
        "task": args.task,
        "num_episodes": args.num_episodes,
        "success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "successes": successes,
    }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Offline evaluation on recorded demonstrations
# ──────────────────────────────────────────────────────────────────────────────

def eval_offline(args: argparse.Namespace, model) -> dict:
    """
    Compute per-step action prediction MSE on held-out HDF5 demonstrations.
    No simulation required — useful for quick iteration.
    """
    import h5py
    from training.datasets.robocasa_dataset import RoboCasaDataset, IGNORE_INDEX

    assert args.data_root is not None, "--data_root is required for offline evaluation"

    # Load normalization stats used by this model
    try:
        action_norm_stats = model.norm_stats[args.unnorm_key]["action"]
    except (KeyError, TypeError):
        action_norm_stats = None
        print(f"Warning: no norm_stats for key '{args.unnorm_key}'; using dataset statistics.")

    dataset = RoboCasaDataset(
        data_root=args.data_root,
        image_transform=model.vision_backbone.get_image_transform(),
        tokenizer=model.llm_backbone.get_tokenizer(),
        prompt_builder_fn=model.llm_backbone.prompt_builder_fn,
        future_action_window_size=args.future_action_window_size,
        past_action_window_size=0,
        camera_name=args.camera_name,
        action_norm_stats=action_norm_stats,
    )

    indices = list(range(min(args.num_episodes * 50, len(dataset))))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    indices = indices[:min(args.num_episodes * 50, len(indices))]

    mse_values = []
    for idx in tqdm(indices, desc="Offline eval"):
        sample = dataset[idx]
        pil_image = Image.fromarray(
            (sample["pixel_values"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        )

        # Recover instruction from input_ids (via decode)
        tokenizer = model.llm_backbone.get_tokenizer()
        ids = sample["input_ids"]
        prompt_text = tokenizer.decode(ids[ids != tokenizer.pad_token_id], skip_special_tokens=True)

        # Extract instruction from the decoded prompt
        marker = "robot take to "
        if marker in prompt_text:
            instruction = prompt_text.split(marker)[-1].rstrip("?").strip()
        else:
            instruction = "perform the task"

        with torch.no_grad():
            predicted_actions, _ = model.predict_action(
                image=pil_image,
                instruction=instruction,
                unnorm_key=args.unnorm_key,
                cfg_scale=args.cfg_scale,
                use_ddim=args.use_ddim,
                num_ddim_steps=args.num_ddim_steps,
                do_sample=False,
            )

        gt_actions = sample["actions"].numpy()  # [W+1, action_dim], normalized
        # Re-normalize predicted for comparison (convert back to normalized space)
        q01 = np.array(dataset.action_norm_stats["q01"])
        q99 = np.array(dataset.action_norm_stats["q99"])
        pred_norm = 2.0 * (predicted_actions - q01) / (q99 - q01 + 1e-8) - 1.0
        pred_norm = np.clip(pred_norm, -1, 1)

        mse = np.mean((pred_norm[:len(gt_actions)] - gt_actions[:len(pred_norm)]) ** 2)
        mse_values.append(float(mse))

    results = {
        "mode": "offline",
        "num_samples": len(mse_values),
        "mean_action_mse": float(np.mean(mse_values)),
        "std_action_mse": float(np.std(mse_values)),
    }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"Loading model from: {args.model_path}")
    model = load_model(args)

    if args.mode == "online":
        results = eval_online(args, model)
    else:
        results = eval_offline(args, model)

    # Print summary
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        if not isinstance(v, list):
            print(f"  {k}: {v}")

    # Save JSON
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.model_path) / "eval" if Path(args.model_path).is_dir() else Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.task}_{args.mode}" if args.mode == "online" else "offline"
    out_file = out_dir / f"results_{tag}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
