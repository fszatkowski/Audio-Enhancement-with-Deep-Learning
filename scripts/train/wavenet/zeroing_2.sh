#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

DATASET_SIZES="256 512 1024 2048 4096 0"

for ds_size in ${DATASET_SIZES}; do
  python3 src/wavenet/train.py \
  --epochs 10 \
  --patience 3 \
  --batch_size 4 \
  \
  --train_files ${ds_size} \
  --transformations zero_002 \
  \
  --learning_rate 0.001 \
  --stack_size 4 \
  --stack_layers 5 \
  \
  --input_kernel_size 31 \
  --residual_channels 16 \
  --skip_kernel_size 31 \
  --skip_channels 16 \
  --model_dir "models/wavenet/zero_002_noise_model_${ds_size}"
done;