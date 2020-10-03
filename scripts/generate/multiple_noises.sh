#!/bin/bash

source scripts/common.sh

function generate_results() {
  pattern=$1
  output=$2

  mkdir -p ${output}

  python src/analysis/mixed/generate_table.py \
    --pattern "${pattern}" \
    --output "${output}results.csv" \
    --average

  python src/analysis/mixed/generate_table.py \
    --pattern "${pattern}" \
    --output "${output}results_no_avg.csv" \


  python src/analysis/mixed/plot.py \
    --pattern "${pattern}" \
    --loss "val" \
    --output "${output}val_log.png" \
    --logx \
    --logy

  python src/analysis/mixed/plot.py \
    --pattern "${pattern}" \
    --loss "val" \
    --output "${output}val_abs.png" \
    --logx

  python src/analysis/mixed/plot.py \
    --pattern "${pattern}" \
    --loss "test" \
    --output "${output}test_log.png" \
    --logx \
    --logy

  python src/analysis/mixed/plot.py \
    --pattern "${pattern}" \
    --loss "test" \
    --output "${output}test_abs.png" \
    --logx
}

generate_results "models/**/mix_model*/metadata.json" "results/mixed/"
generate_results "models/**/mix_weaker*/metadata.json" "results/mixed_weaker_zero/"
generate_results "models/**/mix_no_zero*/metadata.json" "results/mixed_no_zero/"

for set in "test" "val"; do
  python src/analysis/mixed/generate_compare_table.py \
    --pattern "models/**/mix*/metadata.json" \
    --output "results/mixed/${set}_comp.csv" \
    --files 1024 \
    --set "${set}" \

  python src/analysis/mixed/generate_compare_table.py \
    --pattern "models/**/mix*/metadata.json" \
    --output "results/mixed/${set}_log_comp.csv" \
    --files 1024 \
    --set "${set}" \
    --log
done;