#!/bin/bash

root=$(git rev-parse --show-toplevel)
cd ${root}

export "PYTHONPATH=${root}/src"

python3 src/autoencoder/overfit.py
python3 src/wavenet/overfit.py
python3 src/segan/overfit_diff_l1.py