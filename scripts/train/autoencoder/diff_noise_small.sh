#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

NOISE_TYPES="none zero zero_001 zero_002 white_part gaussian_part white_uni gaussian_uni"
SEEDS="123 997 2112"

for seed in ${SEEDS}; do
  for noise in ${NOISE_TYPES}; do
    python3 src/autoencoder/train.py \
      --epochs 10 \
      --patience 3 \
      --batch_size 4 \
      --train_files 256 \
      --transformations ${noise} \
      --learning_rate 0.001 \
      --num_layers 7 \
      --channels 2 \
      --kernel_size 15 \
      --random_seed ${seed} \
      --model_dir "models/autoencoder/small_model_seed_${seed}_${noise}"
  done
done
