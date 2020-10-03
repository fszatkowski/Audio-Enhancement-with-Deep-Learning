#!/bin/bash

source scripts/common.sh

PATTERN="models/autoencoder/final*/metadata.json"
OUTPUT_DIR="results/final/"

mkdir -p ${OUTPUT_DIR}

python src/analysis/final/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}results.csv" \
  --average

python src/analysis/final/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}results_no_avg.csv"
