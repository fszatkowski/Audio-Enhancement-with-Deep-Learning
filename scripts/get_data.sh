#!/bin/bash

root=$(git rev-parse --show-toplevel)
cd ${root}
output_dir="data/fma"

mkdir -p ${output_dir}
curl https://os.unil.cloud.switch.ch/fma/fma_small.zip -o ${output_dir}
unzip ${output_dir}/fma_small
