#!/bin/bash

root=$(git rev-parse --show-toplevel)
cd ${root}
output_dir="data/fma/"

mkdir -p ${output_dir}

zip_file="${output_dir}/fma_small.zip"
curl https://os.unil.cloud.switch.ch/fma/fma_small.zip -o ${zip_file}
unzip ${zip_file} -d "${output_dir}"
