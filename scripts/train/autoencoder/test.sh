#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

SEEDS=${1-"111 1111 11111"}

for seed in ${SEEDS}; do
    python3 src/autoencoder/train.py \
      --epochs 2 \
      --warmup_epochs 2 \
      --patience 4 \
      --batch_size 64 \
      --train_files 64 \
      --test_files 64 \
      --val_files 64 \
      --save_every_n_steps 100 \
      --transformations "default" \
      --max_transformations_applied 2 \
      --learning_rate 0.001 \
      --num_layers 5 \
      --channels 16 \
      --kernel_size 15 \
      --random_seed ${seed} \
      --activation "prelu" \
      --norm batch_norm \
      --model_dir "models/autoencoder/test_${seed}"
done
