#!/bin/bash

source scripts/common.sh

PATTERN_MED="models/**/med_model*/metadata.json"
PATTERN_SMALL="models/**/small_model_*/metadata.json"
OUTPUT_PREFIX="results/diff_noise_medium/diff_noise_medium_models_"

mkdir -p results/diff_noise_medium

python3 src/analysis/diff_noise/plot.py -p "${PATTERN_MED}" -o "${OUTPUT_PREFIX}_val" -l "val"
python3 src/analysis/diff_noise/plot.py -p "${PATTERN_MED}" -o "${OUTPUT_PREFIX}_val_abs" -l "val" --abs_loss
python3 src/analysis/diff_noise/plot.py -p "${PATTERN_MED}" -o "${OUTPUT_PREFIX}_test" -l "test"
python3 src/analysis/diff_noise/plot.py -p "${PATTERN_MED}" -o "${OUTPUT_PREFIX}_test_abs" -l "test" --abs_loss

python3 src/analysis/diff_noise/generate_table.py -p "${PATTERN_MED}" -o "${OUTPUT_PREFIX}val.csv" -l val
python3 src/analysis/diff_noise/generate_table.py -p "${PATTERN_MED}" -o "${OUTPUT_PREFIX}test.csv" -l test

python3 src/analysis/diff_noise/compare.py -f "${PATTERN_MED}" -s "${PATTERN_SMALL}" -o "${OUTPUT_PREFIX}_diff_val.csv" -l val
python3 src/analysis/diff_noise/compare.py -f "${PATTERN_MED}" -s "${PATTERN_SMALL}" -o "${OUTPUT_PREFIX}_diff_test.csv" -l test