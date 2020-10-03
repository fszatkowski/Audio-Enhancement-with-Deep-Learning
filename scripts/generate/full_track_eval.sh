#!/bin/bash

root=$(git rev-parse --show-toplevel)
cd ${root}

export "PYTHONPATH=${root}/src"

MODEL_PREFIX="models/autoencoder/final_997_"
MODEL_SUFFIX="/metadata.json"
NOISE_MODELS=${1-"mix gaussian_uni white_uni zero none gaussian_part white_part zero_001 zero_002"}
WEIGHTS=${2-"uniform tri tz_1 tz_2"}
EVAL_NOISES=${3-"none gaussian_part gaussian_uni white_part white_uni zero_001 zero_002 zero_005"}

OUTPUT_DIR="results/full_track_eval/"
mkdir -p ${OUTPUT_DIR}

for model_noise in ${NOISE_MODELS}; do
  model_path="${MODEL_PREFIX}${model_noise}${MODEL_SUFFIX}"
  for weight in ${WEIGHTS}; do
    for eval_noise in ${EVAL_NOISES}; do
      (
        output_file="${OUTPUT_DIR}${model_noise}_${weight}_${eval_noise}"
        if [ -f "${output_file}" ]; then
          echo "${output_file} already exists, skipping."
          continue
        fi

        python src/eval/full_track_evaluation.py \
          --metadata "${model_path}" \
          --output "${output_file}" \
          --transformations "${eval_noise}" \
          --weights "${weight}" \
          --batch-size 256
      ) &
    done;
  done;
done;