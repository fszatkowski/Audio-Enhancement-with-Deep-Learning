#!/bin/bash

root=$(git rev-parse --show-toplevel)
cd ${root}

export "PYTHONPATH=${root}/src"

MODELS="autoencoder segan wavenet"
TYPES="base_model zero_002_noise_model"

for type in ${TYPES}; do
  for model in ${MODELS}; do
    python3 src/analysis/plot_loss_against_n_files.py -p "models/${model}/${type}*/metadata.json" -o "results/${model}/${type}"
    python3 src/analysis/plot_loss_against_n_files.py -p "models/${model}/${type}*/metadata.json" -o "results/${model}/${type}_logx" --logx
    python3 src/analysis/plot_loss_against_n_files.py -p "models/${model}/${type}*/metadata.json" -o "results/${model}/${type}_logx_absy" --logx --logy
  done;
    python3 src/analysis/plot_loss_against_n_files.py -p "models/**/${type}*/metadata.json" -o "results/all_models_${type}"
    python3 src/analysis/plot_loss_against_n_files.py -p "models/**/${type}*/metadata.json" -o "results/all_models_${type}_logx" --logx
    python3 src/analysis/plot_loss_against_n_files.py -p "models/**/${type}*/metadata.json" -o "results/all_models_${type}_logx_absy" --logx --logy
done;