# CogACT × RoboCasa365

End-to-end guide for evaluating and fine-tuning CogACT on the
[RoboCasa365](https://robocasa.ai/) benchmark — from a fresh machine to reported numbers.

---

## Contents
1. [Environment setup](#1-environment-setup)
2. [Install RoboCasa](#2-install-robocasa)
3. [Download the pretrained model](#3-download-the-pretrained-model)
4. [Download the dataset](#4-download-the-dataset)
5. [Compute normalization statistics](#5-compute-normalization-statistics)
6. [Zero-shot evaluation (pretrained weights)](#6-zero-shot-evaluation-pretrained-weights)
7. [Fine-tuning on RoboCasa](#7-fine-tuning-on-robocasa)
8. [Evaluation of the fine-tuned model](#8-evaluation-of-the-fine-tuned-model)
9. [Understanding the results](#9-understanding-the-results)

---

## 1. Environment setup

> Tested on Ubuntu 20.04/22.04, CUDA 12.1+, Python 3.10.

```bash
# Create and activate conda environment
conda create -n cogact python=3.10 -y
conda activate cogact

# Clone this repo
git clone https://github.com/akira398/CogACT.git
cd CogACT

# Install CogACT and its dependencies
pip install -e .

# Install Flash Attention (required for training; skip for eval-only)
pip install packaging ninja
ninja --version   # should print a version and exit 0
pip install "flash-attn==2.5.5" --no-build-isolation
```

---

## 2. Install RoboCasa

```bash
# Install robosuite (robot simulation backend)
pip install robosuite

# Clone and install RoboCasa v1.0 (RoboCasa365)
mkdir -p third_party
git clone https://github.com/robocasa/robocasa.git third_party/robocasa
cd third_party/robocasa
pip install -e .

# Set up macros / system paths (creates macros_private.py)
python robocasa/scripts/setup_macros.py

# Download kitchen assets: textures, objects, fixtures (~10 GB)
python -m robocasa.scripts.download_kitchen_assets

cd ../..
```

---

## 3. Download the pretrained model

Downloads `config.json`, `dataset_statistics.json`, and the checkpoint `.pt` file
from the Hugging Face Hub.

```bash
# CogACT-Base (~30 GB) — recommended
python scripts/download_pretrained_cogact.py \
    --model_id CogACT/CogACT-Base \
    --save_dir pretrained/CogACT-Base

# Other sizes:
#   --model_id CogACT/CogACT-Small   (~10 GB)
#   --model_id CogACT/CogACT-Large   (~30 GB, larger DiT)
```

After this you will have:
```
pretrained/CogACT-Base/
├── config.json
├── dataset_statistics.json
└── checkpoints/
    └── step-XXXXXX-epoch-XX-loss=X.XXXX.pt
```

---

## 4. Download the dataset

RoboCasa365 uses its own built-in download script (`robocasa.scripts.download_datasets`).
Data is split into two roles:

| Split | Use | Size |
|---|---|---|
| `target` | Evaluation norm-stats + fine-tuning on eval tasks | ~193 h, 500 demos/task |
| `pretrain` | Large-scale multi-task pre-training | ~482 h human + 1,615 h synthetic |

> **For evaluation only (Steps 5–6) you only need `--split target`.**  
> The simulator itself runs from installed RoboCasa — no data needed at eval time.  
> Data is only required to compute action normalization statistics (Step 5).

### Download the 50 evaluation task demos (target split)

This downloads human demos for all 50 RoboCasa365 evaluation tasks
(18 atomic-seen + 16 composite-seen + 16 composite-unseen):

```bash
# The script prompts "Proceed? (y/n)" — pipe `y` to bypass it
echo "y" | python -m robocasa.scripts.download_datasets \
    --split target \
    --source human
```

Data lands in `datasets/` by default (set `DATASET_BASE_PATH` in
`third_party/robocasa/robocasa/macros_private.py` to change it).

To download only a specific subset (e.g., just the 18 atomic-seen tasks for a quick start):

```bash
echo "y" | python -m robocasa.scripts.download_datasets \
    --split target --source human \
    --tasks \
        CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug \
        NavigateKitchen OpenCabinet OpenDrawer OpenStandMixerHead \
        PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceDrawerToCounter \
        PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack \
        TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet
```

Preview what would be downloaded without actually downloading (no prompt for dryrun):

```bash
python -m robocasa.scripts.download_datasets --split target --source human --dryrun
```

### Download pretraining data (for fine-tuning only)

```bash
# Human pretraining demos — 300 tasks, ~482 h
echo "y" | python -m robocasa.scripts.download_datasets --split pretrain --source human

# Synthetic MimicGen demos — 60 atomic tasks, ~1,615 h (optional, very large)
echo "y" | python -m robocasa.scripts.download_datasets --split pretrain --source mimicgen
```

---

## 5. Compute normalization statistics

The pretrained CogACT model was trained on Open-X Embodiment data.
To evaluate zero-shot on RoboCasa, we need to supply action normalization
statistics computed from RoboCasa demonstrations.

RoboCasa365 v1.0 data is downloaded in **LeRobot Parquet format** (not HDF5).
The script auto-detects the format.

```bash
# Point --data_root at the downloaded target split directory:
python scripts/compute_robocasa_stats.py \
    --data_root datasets/robocasa/v1.0/target \
    --output_path data/robocasa/dataset_statistics.json \
    --key robocasa
```

> `datasets/` is the default download location set by
> `robocasa/scripts/setup_macros.py`. Adjust if you changed `DATASET_BASE_PATH`.

**Run on a partial download** (e.g., while the rest is still downloading):

```bash
python scripts/compute_robocasa_stats.py \
    --data_root datasets/robocasa/v1.0/target \
    --tasks PickPlaceCounterToCabinet TurnOnMicrowave TurnOnSinkFaucet \
    --output_path data/robocasa/dataset_statistics.json \
    --key robocasa
```

This scans all `.parquet` files under the data root, computes per-dimension
q1/q99 quantiles over actions, and writes:

```json
{
  "robocasa": {
    "action": {
      "q01": [...],
      "q99": [...],
      "mask": [true, true, true, true, true, true, true]
    }
  }
}
```

Re-running overwrites the file; you can recompute with more data at any time.

---

## 6. Zero-shot evaluation (pretrained weights)

Runs the **exact RoboCasa365 evaluation protocol**:
- 50 tasks (18 atomic-seen + 16 composite-seen + 16 composite-unseen)
- 5 fixed evaluation scenes per task × 10 trials = **50 trials per task**
- Object instance split "B" (held-out from training)
- Reports per-task, per-split, and overall success rate

```bash
python scripts/eval_robocasa365.py \
    --model_path pretrained/CogACT-Base \
    --norm_stats_path data/robocasa/dataset_statistics.json \
    --unnorm_key robocasa \
    --action_model_type DiT-B \
    --trials_per_scene 10 \
    --task_set atomic_seen composite_seen composite_unseen \
    --robot Panda \
    --output_dir results/cogact-base-zero-shot
```

**Quick smoke-test** (atomic tasks only, 2 trials per scene = 10 trials per task):
```bash
python scripts/eval_robocasa365.py \
    --model_path pretrained/CogACT-Base \
    --norm_stats_path data/robocasa/dataset_statistics.json \
    --unnorm_key robocasa \
    --action_model_type DiT-B \
    --trials_per_scene 2 \
    --task_set atomic_seen \
    --robot Panda \
    --output_dir results/cogact-base-smoke-test \
    --record_n_videos 2 \
    --gt_data_root datasets/robocasa/v1.0/target
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--trials_per_scene` | 10 | Trials per (layout, style) scene. Full protocol = 10 (→ 50 total/task). |
| `--task_set` | all 3 | Which splits to evaluate. |
| `--robot` | `Panda` | `Panda` (7-DoF, matches CogACT) or `PandaMobile` (12-DoF). |
| `--action_exec_horizon` | 1 | Steps to execute per predicted chunk before re-predicting. |
| `--cfg_scale` | 1.5 | Classifier-free guidance scale (1.0 = disabled). |
| `--use_ddim` | True | Use DDIM sampling (faster). |
| `--num_ddim_steps` | 10 | DDIM steps. |
| `--record_n_videos` | 0 | Record N randomly chosen trials as MP4 after eval completes. |
| `--gt_data_root` | None | Path to downloaded demo data for ground-truth replay videos. |
| `--video_fps` | 10 | FPS for saved videos. |

Results are saved to `--output_dir/results.json` and `summary.csv`.
Videos (if requested) are saved to `--output_dir/videos/` as:
- `<task>_scene<N>_trial<N>_policy_[success|fail].mp4` — model rollout
- `<task>_scene<N>_trial<N>_GT.mp4` — human demo replay

Requires `imageio[ffmpeg]` (or OpenCV as fallback): `pip install imageio[ffmpeg]`

---

## 7. Fine-tuning on RoboCasa

### Freeze modes

| Mode | What trains | GPU memory | Use when |
|---|---|---|---|
| `dit_only` | DiT action head only | ~20 GB (1× A100) | Quick adaptation, single task |
| `llm_frozen` | Vision backbone + DiT | ~40 GB (1× A100) | Moderate adaptation |
| `full` | All parameters | 8× A100 recommended | Full fine-tune |

### Single-task fine-tuning (e.g., PnP Counter→Cabinet)

```bash
# 1 GPU — freeze everything except the DiT action head
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    scripts/finetune_robocasa.py \
    --pretrained_model pretrained/CogACT-Base \
    --data_root data/robocasa/PickPlaceCounterToCabinet \
    --freeze_mode dit_only \
    --epochs 20 \
    --global_batch_size 32 \
    --per_device_batch_size 32 \
    --learning_rate 2e-5 \
    --run_id cogact-pnp-dit-only \
    --run_root_dir runs/robocasa
```

### Multi-task fine-tuning (all 18 atomic tasks)

```bash
# 8 GPUs — freeze LLM, train vision backbone + DiT
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    scripts/finetune_robocasa.py \
    --pretrained_model pretrained/CogACT-Base \
    --data_root data/robocasa \
    --freeze_mode llm_frozen \
    --epochs 10 \
    --global_batch_size 256 \
    --per_device_batch_size 32 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --run_id cogact-robocasa-multitask \
    --run_root_dir runs/robocasa \
    --wandb_project cogact-robocasa \
    --save_interval 500
```

### Full fine-tune (all parameters)

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    scripts/finetune_robocasa.py \
    --pretrained_model pretrained/CogACT-Base \
    --data_root data/robocasa \
    --freeze_mode full \
    --epochs 10 \
    --global_batch_size 256 \
    --per_device_batch_size 16 \
    --learning_rate 2e-5 \
    --run_id cogact-robocasa-full-finetune \
    --run_root_dir runs/robocasa
```

Checkpoints and `dataset_statistics.json` are saved under:
```
runs/robocasa/<run_id>/
├── config.json
├── config.yaml
├── dataset_statistics.json        ← used at eval time
└── checkpoints/
    └── step-XXXXXX-epoch-XX-loss=X.XXXX.pt
```

---

## 8. Evaluation of the fine-tuned model

After fine-tuning, run the same RoboCasa365 protocol.
The `dataset_statistics.json` in the run directory is loaded automatically.

```bash
python scripts/eval_robocasa365.py \
    --model_path runs/robocasa/cogact-robocasa-multitask \
    --unnorm_key robocasa \
    --action_model_type DiT-B \
    --trials_per_scene 10 \
    --task_set atomic_seen composite_seen composite_unseen \
    --robot Panda \
    --output_dir results/cogact-robocasa-multitask-eval
```

---

## 9. Understanding the results

### Baseline numbers from the RoboCasa365 paper

| Method | Atomic-Seen | Composite-Seen | Composite-Unseen |
|---|---|---|---|
| Diffusion Policy | 15.7% | 0.2% | 1.25% |
| π₀ | 36.3% | 5.2% | 0.7% |
| π₀.5 | 39.6% | 7.1% | 1.2% |
| GR00T N1.5 | **43.0%** | **9.6%** | **4.4%** |
| GR00T N1.5 (pretrain + full target data) | 68.5% | 40.6% | 42.1% |

### Notes on zero-shot CogACT results
- CogACT was pretrained on **Open-X Embodiment** (no RoboCasa data).
- `NavigateKitchen` requires a mobile base — will fail with `--robot Panda`.
- Composite task performance depends heavily on the action chunking horizon
  (`--action_exec_horizon`); tuning this can have a significant effect.
- For reproducibility, use `--seed 0` (default) and report
  `trials_per_scene=10` (50 trials/task).

### Output files

```
results/<run_name>/
├── results.json     ← full per-task, per-scene breakdown
└── summary.csv      ← one line per task + per-split averages
```
