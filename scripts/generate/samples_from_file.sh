#!/bin/bash

source scripts/common.sh

NOISE_TYPES="zero zero_001 zero_002 white_part gaussian_part white_uni gaussian_uni"
input_file=$1

for noise in ${NOISE_TYPES}; do
    python src/inference/create_denoising_samples.py \
      -m models/autoencoder/final_997_mix/metadata.json \
      -i ${input_file} \
      -o test_dir/${input_file:23:6} \
      -n ${noise} \
      --sr 22050
done