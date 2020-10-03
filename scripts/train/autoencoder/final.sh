#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

SEEDS=${1-"997"}
NOISES=${2-"mix gaussian_uni white_uni zero none gaussian_part white_part zero_001 zero_002"}

for seed in ${SEEDS}; do
  for noise in ${NOISES}; do
      python3 src/autoencoder/train.py \
        --epochs 20 \
        --patience 3 \
        --batch_size 16 \
        --train_files 512 \
        --save_every_n_steps 100 \
        --transformations ${noise} \
        --learning_rate 0.001 \
        --num_layers 5 \
        --channels 16 \
        --kernel_size 15 \
        --random_seed ${seed} \
        --activation "prelu" \
        --norm batch_norm \
        --model_dir "models/autoencoder/final_${seed}_${noise}"
    done
done
