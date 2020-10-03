#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

SEEDS=${1-"123 2112 997"}
BATCH_SIZES=${2-"64 32"}
LEARNING_RATES=${3-"0.01 0.0001"}


for seed in ${SEEDS}; do
  for bs in ${BATCH_SIZES}; do
    for lr in ${LEARNING_RATES}; do
      python3 src/autoencoder/train.py \
        --epochs 20 \
        --patience 3 \
        --batch_size ${bs} \
        --train_files 512 \
        --save_every_n_steps 100 \
        --transformations none \
        --learning_rate ${lr} \
        --num_layers 5 \
        --channels 16 \
        --kernel_size 15 \
        --random_seed ${seed} \
        --activation "prelu" \
        --norm batch_norm \
        --model_dir "models/autoencoder/trainhyper_${seed}_lr_${lr}_b_${bs}"
    done
  done
done
