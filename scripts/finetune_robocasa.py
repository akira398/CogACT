"""
finetune_robocasa.py

Finetune a pretrained CogACT model on RoboCasa demonstrations.

Run (single GPU):
    torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/finetune_robocasa.py \
        --pretrained_model CogACT/CogACT-Base \
        --data_root data/robocasa/PnPCounterToCab \
        --run_id cogact-robocasa-pnp

Run (multi-GPU, e.g. 8 GPUs):
    torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/finetune_robocasa.py \
        --pretrained_model CogACT/CogACT-Base \
        --data_root data/robocasa/PnPCounterToCab \
        --global_batch_size 64 \
        --per_device_batch_size 8

Freeze options (from most to least frozen):
    --freeze_mode dit_only      Train only the DiT action head  [default, fastest]
    --freeze_mode llm_frozen    Train DiT + vision backbone
    --freeze_mode full          Train everything (full finetune)
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from training import VLAMetrics, get_train_strategy
from training.datasets import RoboCasaDataset, RoboCasaCollator
from vla import load_vla

os.environ["TOKENIZERS_PARALLELISM"] = "false"

overwatch = initialize_overwatch(__name__)

# Mapping from freeze_mode to the stage string used by CogACT.freeze_backbones()
_FREEZE_MODE_TO_STAGE = {
    "dit_only": "align",              # freeze vision + LLM, only DiT trains
    "llm_frozen": "finetune",         # freeze LLM only
    "full": "full-finetune",          # nothing frozen
}


@dataclass
class RoboCasaFinetuneConfig:
    # fmt: off

    # === Model ===
    pretrained_model: str = "CogACT/CogACT-Base"
    # Local path to a .pt checkpoint OR a HuggingFace Hub repo ID.
    # Examples:
    #   "CogACT/CogACT-Base"  (downloads from HF)
    #   "pretrained/CogACT-Base/checkpoints/step-000070000-epoch-00-loss=0.0448.pt"

    action_model_type: str = "DiT-B"
    future_action_window_size: int = 15
    past_action_window_size: int = 0
    action_dim: int = 7

    # === Freeze strategy ===
    freeze_mode: str = "dit_only"
    # Choices: "dit_only" | "llm_frozen" | "full"
    #   dit_only   — freeze vision backbone + LLM; train only the DiT head (fastest)
    #   llm_frozen — freeze LLM only; train vision backbone + DiT
    #   full       — train all parameters (requires most GPU memory)

    # === Data ===
    data_root: str = "data/robocasa"
    # Path to a directory containing .hdf5 files, or a single .hdf5 file.
    camera_name: str = "robot0_agentview_left"

    # === Run / logging ===
    run_root_dir: Path = Path("runs/robocasa")
    run_id: Optional[str] = None
    save_interval: int = 1000
    seed: int = 42
    trackers: Tuple[str, ...] = ("jsonl", "wandb")
    wandb_project: str = ""
    wandb_entity: str = ""

    # === Optimization ===
    epochs: int = 10
    max_steps: Optional[int] = None
    global_batch_size: int = 64
    per_device_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear-warmup+cosine-decay"
    warmup_ratio: float = 0.05
    repeated_diffusion_steps: int = 8

    # === Distributed / mixed precision ===
    train_strategy: str = "fsdp-full-shard"
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = True

    # === HF credentials ===
    hf_token: Union[str, Path] = Path(".hf_token")

    # fmt: on

    def __post_init__(self) -> None:
        assert self.freeze_mode in _FREEZE_MODE_TO_STAGE, (
            f"freeze_mode must be one of {list(_FREEZE_MODE_TO_STAGE)}; got '{self.freeze_mode}'"
        )
        assert self.global_batch_size % self.per_device_batch_size == 0


@draccus.wrap()
def finetune(cfg: RoboCasaFinetuneConfig) -> None:
    overwatch.info("CogACT RoboCasa Finetuning :: Starting")

    # Distributed setup (torchrun sets RANK, LOCAL_RANK, WORLD_SIZE automatically)
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Build run identifier and directories
    if cfg.run_id is None:
        model_tag = cfg.pretrained_model.split("/")[-1]
        cfg.run_id = f"cogact-robocasa-{model_tag}-freeze={cfg.freeze_mode}"
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(run_dir / "checkpoints", exist_ok=True)

    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)

    # Resolve HF token
    if isinstance(cfg.hf_token, Path) and cfg.hf_token.exists():
        hf_token = cfg.hf_token.read_text().strip()
    elif isinstance(cfg.hf_token, str):
        hf_token = os.environ.get(cfg.hf_token, "")
    else:
        hf_token = None

    # Save config (rank 0 only)
    if overwatch.is_rank_zero():
        import yaml
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            json.dump(yaml.safe_load(f_yaml), f_json, indent=2)

    dist.barrier()

    # ------------------------------------------------------------------ #
    # 1. Load pretrained CogACT                                          #
    # ------------------------------------------------------------------ #
    overwatch.info(f"Loading pretrained CogACT from '{cfg.pretrained_model}'")
    vla = load_vla(
        cfg.pretrained_model,
        hf_token=hf_token,
        load_for_training=True,
        action_model_type=cfg.action_model_type,
        action_dim=cfg.action_dim,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
        use_ema=False,
    )

    for param in vla.parameters():
        assert param.dtype == torch.float32, "Model must be in FP32 at load time"

    # ------------------------------------------------------------------ #
    # 2. Freeze parameters according to selected freeze_mode             #
    # ------------------------------------------------------------------ #
    stage = _FREEZE_MODE_TO_STAGE[cfg.freeze_mode]
    overwatch.info(f"Freezing backbones with stage='{stage}' (freeze_mode='{cfg.freeze_mode}')")
    vla.freeze_backbones(stage)

    n_total = sum(p.numel() for p in vla.parameters())
    n_trainable = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(
        f"Parameters: {n_total / 1e6:.1f}M total, {n_trainable / 1e6:.1f}M trainable"
    )

    # ------------------------------------------------------------------ #
    # 3. Build RoboCasa dataset + collator                               #
    # ------------------------------------------------------------------ #
    overwatch.info(f"Building RoboCasa dataset from '{cfg.data_root}'")
    dataset = RoboCasaDataset(
        data_root=cfg.data_root,
        image_transform=vla.vision_backbone.get_image_transform(),
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
        camera_name=cfg.camera_name,
    )
    overwatch.info(f"Dataset size: {len(dataset)} samples")

    # Save dataset statistics for inference-time un-normalisation
    if overwatch.is_rank_zero():
        save_dataset_statistics(dataset.dataset_statistics, run_dir)

    dist.barrier()

    collator = RoboCasaCollator(
        pad_token_id=vla.llm_backbone.get_tokenizer().pad_token_id
    )

    # ------------------------------------------------------------------ #
    # 4. Training strategy (FSDP)                                        #
    # ------------------------------------------------------------------ #
    grad_accumulation_steps = cfg.global_batch_size // cfg.per_device_batch_size // overwatch.world_size()

    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vla,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(dataset))

    # ------------------------------------------------------------------ #
    # 5. DataLoader with DistributedSampler                              #
    # ------------------------------------------------------------------ #
    sampler = DistributedSampler(
        dataset,
        num_replicas=overwatch.world_size(),
        rank=overwatch.rank(),
        shuffle=True,
        seed=cfg.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    # ------------------------------------------------------------------ #
    # 6. Metrics                                                          #
    # ------------------------------------------------------------------ #
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=None,
        resume_epoch=None,
    )

    # ------------------------------------------------------------------ #
    # 7. Training loop                                                    #
    # ------------------------------------------------------------------ #
    overwatch.info("Starting RoboCasa finetuning loop")

    steps_per_epoch = len(dataloader) // grad_accumulation_steps
    total_steps = steps_per_epoch * cfg.epochs if cfg.max_steps is None else cfg.max_steps

    status = metrics.get_status()
    with tqdm(
        total=total_steps,
        desc=status,
        leave=False,
        disable=not overwatch.is_rank_zero(),
    ) as progress:
        for epoch in range(cfg.epochs):
            train_strategy.vlm.train()
            sampler.set_epoch(epoch)
            train_strategy.optimizer.zero_grad()

            for train_idx, batch in enumerate(dataloader):
                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=cfg.enable_mixed_precision_training,
                ):
                    loss, _ = train_strategy.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        actions=batch["actions"],
                        pixel_values=batch["pixel_values"],
                        action_masks=batch["action_masks"],
                        labels=batch["labels"],
                        output_hidden_states=True,
                        repeated_diffusion_steps=cfg.repeated_diffusion_steps,
                    )

                metrics.commit(loss=loss)
                (loss / grad_accumulation_steps).backward()

                if (train_idx + 1) % grad_accumulation_steps == 0:
                    train_strategy.clip_grad_norm()
                    train_strategy.optimizer.step()
                    train_strategy.lr_scheduler.step()
                    train_strategy.optimizer.zero_grad()

                    metrics.commit(
                        update_step_time=True,
                        global_step=metrics.global_step + 1,
                        epoch=epoch,
                        lr=train_strategy.lr_scheduler.get_last_lr()[0],
                    )
                    status = metrics.push()
                    progress.update()
                    progress.set_description(status)

                    if metrics.global_step % cfg.save_interval == 0:
                        train_strategy.save_checkpoint(
                            run_dir, metrics.global_step, epoch, loss.item(),
                            only_trainable=True,
                        )
                        dist.barrier()

                    if cfg.max_steps is not None and metrics.global_step >= cfg.max_steps:
                        train_strategy.save_checkpoint(
                            run_dir, metrics.global_step, epoch, loss.item(),
                            only_trainable=True,
                        )
                        dist.barrier()
                        metrics.finalize()
                        overwatch.info("Reached max_steps — done.")
                        return

            # End-of-epoch checkpoint
            if cfg.max_steps is None:
                train_strategy.save_checkpoint(
                    run_dir, metrics.global_step, epoch, loss.item(),
                    only_trainable=True,
                )
                dist.barrier()

    metrics.finalize()
    overwatch.info("Finetuning complete.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    finetune()
