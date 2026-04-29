"""
download_pretrained_cogact.py

Download a pretrained CogACT checkpoint from the Hugging Face Hub to a local directory.

Available model IDs:
    CogACT/CogACT-Small   (~1B action model on Phi-2 backbone)
    CogACT/CogACT-Base    (~7B LLaMA-2 backbone, DiT-B action model)  [recommended]
    CogACT/CogACT-Large   (~7B LLaMA-2 backbone, DiT-L action model)

Usage:
    python scripts/download_pretrained_cogact.py \
        --model_id CogACT/CogACT-Base \
        --save_dir pretrained/CogACT-Base
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a pretrained CogACT model from HF Hub.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="CogACT/CogACT-Base",
        help="HuggingFace Hub model ID (e.g. CogACT/CogACT-Base).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Local directory to save the model. Defaults to pretrained/<model_name>.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for gated models. Falls back to HF_TOKEN env var or .hf_token file.",
    )
    return parser.parse_args()


def resolve_token(hf_token_arg: str | None) -> str | None:
    if hf_token_arg:
        return hf_token_arg
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    token_file = Path(".hf_token")
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def main() -> None:
    args = parse_args()
    token = resolve_token(args.hf_token)

    model_name = args.model_id.split("/")[-1]
    save_dir = Path(args.save_dir) if args.save_dir else Path("pretrained") / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model_id} → {save_dir}")

    # Validate the repo exists and list its files
    fs = HfFileSystem(token=token)
    if not fs.exists(args.model_id):
        raise ValueError(
            f"Model '{args.model_id}' not found on the Hub. "
            "Check available models: CogACT/CogACT-Small, CogACT/CogACT-Base, CogACT/CogACT-Large"
        )

    # Download everything: config.json, dataset_statistics.json, checkpoints/*.pt
    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(save_dir),
        token=token,
        ignore_patterns=["*.safetensors", "flax_model*", "tf_model*"],
    )

    # Verify the essential files are present
    required = ["config.json", "dataset_statistics.json"]
    for fname in required:
        assert (save_dir / fname).exists(), f"Missing {fname} after download!"

    ckpts = list((save_dir / "checkpoints").glob("*.pt"))
    assert ckpts, "No .pt checkpoint found after download!"

    print(f"\nDownload complete. Files in {save_dir}:")
    for p in sorted(save_dir.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.relative_to(save_dir)}  ({size_mb:.1f} MB)")

    print(f"\nTo finetune on RoboCasa, pass:  --pretrained_model {save_dir}")


if __name__ == "__main__":
    main()
