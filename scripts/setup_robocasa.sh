#!/usr/bin/env bash
# setup_robocasa.sh
#
# Installs RoboCasa and its dependencies, then downloads the dataset.
#
# Usage:
#   bash scripts/setup_robocasa.sh [TASK] [N_DEMOS] [DATA_DIR]
#
#   TASK      RoboCasa task name, e.g. PnPCounterToCab (default: PnPCounterToCab)
#   N_DEMOS   Number of demos to download (default: 1000)
#   DATA_DIR  Where to store the data     (default: data/robocasa)
#
# After this script succeeds you can run:
#   bash scripts/finetune_robocasa.sh

set -euo pipefail

TASK=${1:-"PnPCounterToCab"}
N_DEMOS=${2:-1000}
DATA_DIR=${3:-"data/robocasa"}

echo "=== Installing robosuite ==="
pip install robosuite

echo "=== Cloning and installing RoboCasa ==="
if [ ! -d "third_party/robocasa" ]; then
    mkdir -p third_party
    git clone https://github.com/robocasa/robocasa.git third_party/robocasa
fi
pip install -e third_party/robocasa

echo "=== Downloading RoboCasa assets (tabletop fixtures, textures, ...) ==="
python -c "import robocasa; robocasa.utils.download_assets()"

echo "=== Downloading dataset: task=${TASK}, n_demos=${N_DEMOS} ==="
mkdir -p "${DATA_DIR}/${TASK}"

# RoboCasa ships a download script; invoke it non-interactively.
python third_party/robocasa/robocasa/scripts/download_datasets.py \
    --tasks "${TASK}" \
    --n_demos "${N_DEMOS}" \
    --path "${DATA_DIR}/${TASK}"

echo ""
echo "Done. Data is in: ${DATA_DIR}/${TASK}"
echo "Run finetuning with:"
echo "  torchrun --standalone --nnodes 1 --nproc-per-node <N_GPUS> scripts/finetune_robocasa.py \\"
echo "      --data_root ${DATA_DIR}/${TASK} \\"
echo "      --task_name ${TASK}"
