#!/bin/bash

source scripts/common.sh

MODELS="autoencoder segan wavenet"
PATTERN="base_model*/metadata.json"
AGG_OUT_PATH="results/base/all_models_base"
AE_SEED_PATH="results/base/ae_seeds_plot"

mkdir -p results/base

for model in ${MODELS}; do
    mkdir -p "results/base/${model}"
    python3 src/analysis/base/plot.py -p "models/${model}/${PATTERN}" -o "results/base/${model}/${model}_base"
    python3 src/analysis/base/plot.py -p "models/${model}/${PATTERN}" -o "results//base/${model}/${model}_base_logx" --logx
    python3 src/analysis/base/plot.py -p "models/${model}/${PATTERN}" -o "results/base/${model}/${model}_base_logx_absy" --logx --logy
done;

python3 src/analysis/base/plot.py -p "models/**/${PATTERN}" -o ${AGG_OUT_PATH}
python3 src/analysis/base/plot.py -p "models/**/${PATTERN}" -o "${AGG_OUT_PATH}_logx" --logx
python3 src/analysis/base/plot.py -p "models/**/${PATTERN}" -o "${AGG_OUT_PATH}_logx_absy" --logx --logy

python3 src/analysis/base/plot_diff_seeds.py -o ${AE_SEED_PATH}
python3 src/analysis/base/plot_diff_seeds.py -o "${AE_SEED_PATH}_logx" --logx
python3 src/analysis/base/plot_diff_seeds.py -o "${AE_SEED_PATH}_logx_absy" --logx --logy

python3 src/analysis/base/generate_table.py -o "results/base/base_models.csv"