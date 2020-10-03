#!/bin/bash

source scripts/common.sh

PREFIX="results/overfit/"

mkdir -p ${PREFIX}

python3 src/analysis/overfit/generate_table.py -o "${PREFIX}overfit_models_results.csv"
python3 src/analysis/overfit/plot_l1.py -o "${PREFIX}overfit_l1"
python3 src/analysis/overfit/plot_l1.py -o "${PREFIX}overfit_l1_abs" --abs_loss