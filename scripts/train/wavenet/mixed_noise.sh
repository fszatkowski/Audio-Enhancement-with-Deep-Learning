#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

DATASET_SIZES=${1-"256 512 1024 2048 4096 0"}
SEEDS=${2-"123 2112 997"}

for ds_size in ${DATASET_SIZES}; do
  for seed in ${SEEDS}; do
    python3 src/wavenet/train.py \
    --epochs 10 \
    --patience 3 \
    --batch_size 4 \
    \
    --train_files ${ds_size} \
    --transformations mix \
    \
    --learning_rate 0.001 \
    --stack_size 4 \
    --stack_layers 5 \
    \
    --input_kernel_size 31 \
    --residual_channels 16 \
    --skip_kernel_size 31 \
    --skip_channels 16 \
    --random_seed ${seed} \
    --model_dir "models/wavenet/mix_model_seed_${seed}_${ds_size}"
  done;
done;
