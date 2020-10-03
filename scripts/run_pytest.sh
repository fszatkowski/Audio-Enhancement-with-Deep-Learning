#!/bin/bash

ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="${ROOT}/src"

cd ${ROOT}
pytest
cd -