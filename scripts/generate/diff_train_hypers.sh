#!/bin/bash

source scripts/common.sh

PATTERN="models/autoencoder/trainhyper*/metadata.json"
OUTPUT_DIR="results/train_hypers/"

mkdir -p ${OUTPUT_DIR}

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}short_results.csv" \
  --average \
  --short \
  --activation \
  --trainhypers

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}short_results_no_avg.csv" \
  --short \
  --activation \
  --trainhypers

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}results.csv" \
  --average \
  --activation \
  --trainhypers

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}results_no_avg.csv" \
  --activation \
  --trainhypers
