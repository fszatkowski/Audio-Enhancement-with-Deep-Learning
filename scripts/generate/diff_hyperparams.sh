#!/bin/bash

source scripts/common.sh

PATTERN="models/autoencoder/hiper*/metadata.json"
OUTPUT_DIR="results/ae_hyperparams/"

mkdir -p ${OUTPUT_DIR}

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}short_results.csv" \
  --average \
  --short

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}short_results_no_avg.csv" \
  --short

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}ae_results.csv" \
  --average

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}ae_results_no_avg.csv"

python src/analysis/ae_hypers/plot.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}/val_log.png" \
  --loss val \
  --logx \
  --logy \
  --average

python src/analysis/ae_hypers/plot.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}/val_abs.png" \
  --logx \
  --loss val \
  --average

python src/analysis/ae_hypers/plot.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}/test_log.png" \
  --loss test \
  --logx \
  --logy \
  --average

python src/analysis/ae_hypers/plot.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}/test_abs.png" \
  --logx \
  --loss test \
  --average
