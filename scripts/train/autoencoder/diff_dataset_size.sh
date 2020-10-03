#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

DATASET_SIZES="256 512 1024 2048 4096 0"

for ds_size in ${DATASET_SIZES}; do
  python3 src/autoencoder/train.py \
    --epochs 10 \
    --patience 3 \
    --batch_size 4 \
    \
    --train_files ${ds_size} \
    --transformations none \
    \
    --learning_rate 0.001 \
    --num_layers 7 \
    --channels 2 \
    --kernel_size 15 \
    --model_dir "models/autoencoder/base_model_${ds_size}"
done;