#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

NOISES=${1-"mix_no_zero mix_weaker_zero"}
SEEDS=${2-"123 2112 997"}

for noise in ${NOISES}; do
  for seed in ${SEEDS}; do
    python3 src/segan/train.py \
    --epochs 10 \
    --patience 3 \
    --batch_size 4 \
    \
    --train_files 1024 \
    --transformations ${noise} \
    \
    --g_lr 0.0002 \
    --d_lr 0.0002 \
    --l1_alpha 10 \
    \
    --n_layers 7 \
    --init_channels 2 \
    --kernel_size 15 \
    --random_seed ${seed} \
    --model_dir "models/segan/${noise}_mix_model_seed_${seed}"
  done
done
