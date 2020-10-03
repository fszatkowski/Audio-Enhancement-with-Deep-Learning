#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

NOISE_TYPES="none zero zero_001 zero_002 white_part gaussian_part white_uni gaussian_uni"
SEEDS="123 997 2112"

for seed in ${SEEDS}; do
  for noise in ${NOISE_TYPES}; do
    python3 src/segan/train.py \
      --epochs 10 \
      --patience 3 \
      --batch_size 4 \
      --train_files 1024 \
      --transformations ${noise} \
      --g_lr 0.0002 \
      --d_lr 0.0002 \
      --l1_alpha 10 \
      --n_layers 7 \
      --init_channels 2 \
      --kernel_size 15 \
      --random_seed ${seed} \
      --model_dir "models/segan/med_model_seed_${seed}_${noise}"
  done
done
