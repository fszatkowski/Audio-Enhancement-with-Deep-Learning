#!/bin/bash

source scripts/common.sh

PATTERN_AE="models/autoencoder/hiper*/metadata.json"
PATTERN_BN="models/autoencoder/batch_norm*/metadata.json"
OUTPUT_DIR="results/batch_norm/"

AE_FILE="${OUTPUT_DIR}short_results.csv"
BN_FILE="${OUTPUT_DIR}bn_short_results.csv"

mkdir -p ${OUTPUT_DIR}

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN_AE}" \
  --output "${AE_FILE}" \
  --average \
  --short

python src/analysis/ae_hypers/generate_table.py \
  --pattern "${PATTERN_BN}" \
  --output "${BN_FILE}" \
  --average \
  --short

python src/analysis/batch_norm/generate_table.py \
  --ae "${AE_FILE}" \
  --bn "${BN_FILE}" \
  --output "${OUTPUT_DIR}comparison.csv" \
