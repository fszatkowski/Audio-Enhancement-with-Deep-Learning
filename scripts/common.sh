#!/bin/bash

root=$(git rev-parse --show-toplevel)
cd ${root}

export "PYTHONPATH=${root}/src"

python3 src/analysis/update_results.py