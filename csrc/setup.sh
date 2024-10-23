#!/bin/bash
export CUDACXX=/usr/local/cuda/bin/nvcc

git clone --branch v3.0.0 https://github.com/NVIDIA/cutlass.git

python setup_cuda.py install