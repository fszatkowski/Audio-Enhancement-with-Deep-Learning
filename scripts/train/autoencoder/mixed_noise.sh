#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

DATASET_SIZES=${1-"256 512 1024 2048 4096 0"}
SEEDS=${2-"123 2112 997"}

for ds_size in ${DATASET_SIZES}; do
  for seed in ${SEEDS}; do
    python3 src/autoencoder/train.py \
      --epochs 10 \
      --patience 3 \
      --batch_size 4 \
      --train_files \
      ${ds_size} \
      --transformations mix \
      --learning_rate \
      0.001 \
      --num_layers 7 \
      --channels 2 \
      --kernel_size 15 \
      --random_seed ${seed} \
      --model_dir "models/autoencoder/mix_model_seed_${seed}_${ds_size}"
  done
done
