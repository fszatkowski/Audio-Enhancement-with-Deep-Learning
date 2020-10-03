#!/bin/bash

source scripts/common.sh

PATTERN="models/autoencoder/activations*/metadata.json"
OUTPUT_DIR="results/activations/"

mkdir -p ${OUTPUT_DIR}

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}short_results.csv" \
  --average \
  --short \
  --activation

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}short_results_no_avg.csv" \
  --short \
  --activation

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}results.csv" \
  --average \
  --activation

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}results_no_avg.csv" \
  --activation