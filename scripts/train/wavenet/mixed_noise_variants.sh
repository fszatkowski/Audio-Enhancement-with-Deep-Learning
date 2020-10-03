#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

NOISES=${1-"mix_no_zero mix_weaker_zero"}
SEEDS=${2-"123 2112 997"}

for noise in ${NOISES}; do
  for seed in ${SEEDS}; do
    python3 src/wavenet/train.py \
    --epochs 10 \
    --patience 3 \
    --batch_size 4 \
    \
    --train_files 1024 \
    --transformations ${noise} \
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
    --model_dir "models/wavenet/${noise}_mix_model_seed_${seed}"
  done
done
