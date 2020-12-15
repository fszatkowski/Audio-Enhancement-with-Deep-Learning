#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

LRS=${1-"0.0001 0.00005 0.00001 0.000005 0.000001"}

for lr in ${LRS}; do
    python3 src/autoencoder/train.py \
      --epochs 30 \
      --warmup_epochs 0 \
      --patience 5 \
      --batch_size 32 \
      --train_files 1984 \
      --test_files 32 \
      --val_files 32 \
      --save_every_n_steps 100 \
      --transformations "default" \
      --max_transformations_applied 2 \
      --learning_rate ${lr} \
      --num_layers 7 \
      --channels 16 \
      --kernel_size 15 \
      --random_seed 111 \
      --activation "prelu" \
      --norm batch_norm \
      --model_dir "models/autoencoder/new_no_warmup_lr_${lr}"
done
