#!/bin/bash -e
# Test to ensure sif file is all good.
python -c "import asmcmc; import numpy as np; import ase; import matplotlib; import signac"

mkdir workspace
echo "My hash is $1"
mv $1 workspace    # First argument is the hash

# Copied /usr/bin/unzip in data_subsample, so I have access to it.

echo "Running Python"
python project.py run
