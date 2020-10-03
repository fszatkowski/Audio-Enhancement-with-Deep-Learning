#!/bin/bash

source scripts/common.sh

PATTERN="models/**/small_model*/metadata.json"
OUTPUT_PREFIX="results/diff_noise_small/diff_noise_small_models_"

mkdir -p results/diff_noise_small

python3 src/analysis/diff_noise/plot.py -p "${PATTERN}" -o "${OUTPUT_PREFIX}_val" -l "val"
python3 src/analysis/diff_noise/plot.py -p "${PATTERN}" -o "${OUTPUT_PREFIX}_val_abs" -l "val" --abs_loss
python3 src/analysis/diff_noise/plot.py -p "${PATTERN}" -o "${OUTPUT_PREFIX}_test" -l "test"
python3 src/analysis/diff_noise/plot.py -p "${PATTERN}" -o "${OUTPUT_PREFIX}_test_abs" -l "test" --abs_loss

python3 src/analysis/diff_noise/generate_table.py -p "${PATTERN}" -o "${OUTPUT_PREFIX}val.csv" -l val
python3 src/analysis/diff_noise/generate_table.py -p "${PATTERN}" -o "${OUTPUT_PREFIX}test.csv" -l test
