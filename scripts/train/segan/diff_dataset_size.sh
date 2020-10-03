#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

DATASET_SIZES="256 512 1024 2048 4096 0"

for ds_size in ${DATASET_SIZES}; do
  python3 src/segan/train.py \
  --epochs 10 \
  --patience 3 \
  --batch_size 4 \
  \
  --train_files ${ds_size} \
  --transformations none \
  \
  --g_lr 0.0002 \
  --d_lr 0.0002 \
  --l1_alpha 10 \
  \
  --n_layers 7 \
  --init_channels 2 \
  --kernel_size 15 \
  --model_dir "models/segan/base_model_${ds_size}"
done;