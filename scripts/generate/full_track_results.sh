#!/bin/bash

source scripts/common.sh

PATTERN="results/full_track_eval/*"
OUTPUT_DIR="results/full_track/"
WEIGHTS="Równomierny"

mkdir -p ${OUTPUT_DIR}

python src/analysis/full_track/generate_table.py \
  --output "${OUTPUT_DIR}weights.csv"

# MODEL VS NOISE
python src/analysis/full_track/generate_heatmap.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}heatmap_mse" \
  --weights "${WEIGHTS}" \
  --target "MSE" \
  --title "Błąd średniokwadratowy" \
  --name "model" \
  --lognorm

python src/analysis/full_track/generate_heatmap.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}heatmap_mae" \
  --weights "${WEIGHTS}" \
  --target "MAE" \
  --title "Błąd średni" \
  --name "model" \
  --lognorm

python src/analysis/full_track/generate_heatmap.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}heatmap_snr" \
  --weights "${WEIGHTS}" \
  --target "SNR" \
  --name "model" \
  --title "Stosunek sygnału do szumu (SNR)"

python src/analysis/full_track/generate_heatmap.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}heatmap_psnr" \
  --weights "${WEIGHTS}" \
  --target "PSNR" \
  --name "model" \
  --title "Szczytowy stosunek sygnału do szumu (PSNR)"

# BENCHMARKS
python src/analysis/full_track/generate_comp_heatmap.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}comp_heatmap_mae" \
  --weights "${WEIGHTS}" \
  --target "MAE" \
  --title "MAE - różnica między modelem a bazowym algorytmem"

python src/analysis/full_track/generate_comp_heatmap.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}comp_heatmap_mse" \
  --weights "${WEIGHTS}" \
  --target "MSE" \
  --title "MSE - różnica między modelem a bazowym algorytmem"

python src/analysis/full_track/generate_comp_heatmap.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}comp_heatmap_snr" \
  --weights "${WEIGHTS}" \
  --target "SNR" \
  --title "SNR oraz PSNR - różnica między modelem a bazowym algorytmem"

# BENCHMARKS (RELATIVE)
python src/analysis/full_track/generate_comp_heatmap.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}comp_rel_heatmap_mae" \
  --weights "${WEIGHTS}" \
  --target "MAE" \
  --title "MAE - stosunek wyników modelu do wyników bazowego algorytmu" \
  --relative

python src/analysis/full_track/generate_comp_heatmap.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}comp_rel_heatmap_mse" \
  --weights "${WEIGHTS}" \
  --target "MSE" \
  --title "MSE - stosunek wyników modelu do wyników bazowego algorytmu" \
  --relative

python src/analysis/full_track/generate_comp_heatmap.py \
  --pattern "${PATTERN}" \
  --output "${OUTPUT_DIR}comp_rel_heatmap_snr" \
  --weights "${WEIGHTS}" \
  --target "SNR" \
  --title "SNR oraz PSNR - stosunek wyników modelu do wyników bazowego algorytmu" \
  --relative

