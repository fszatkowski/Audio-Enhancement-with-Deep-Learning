#!/bin/bash

root=$(git rev-parse --show-toplevel)
export PYTHONPATH="${root}/src"
cd ${root}

SEEDS=${1-"123 2112 997"}

ACTIVATIONS="relu elu prelu gelu swish"
LAYERS="5"
KERNELS="15"
CHANNELS="16"

for seed in ${SEEDS}; do
  for layer in ${LAYERS}; do
    for kernel in ${KERNELS}; do
      for channel in ${CHANNELS}; do
         for activation in ${ACTIVATIONS}; do
           python3 src/autoencoder/train.py \
            --epochs 10 \
            --patience 3 \
            --batch_size 16 \
            \
            --train_files 512 \
            --save_every_n_steps 100 \
            --transformations none \
            \
            --learning_rate 0.001 \
            --num_layers ${layer} \
            --channels ${channel} \
            --kernel_size ${kernel} \
            --random_seed ${seed} \
            --activation ${activation} \
            --norm batch_norm \
            --model_dir "models/autoencoder/activations_${seed}_${activation}_l_${layer}_c_${channel}_k_${kernel}"
        done;
      done;
    done;
  done;
done;