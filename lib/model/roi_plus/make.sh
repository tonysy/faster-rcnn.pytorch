#!/usr/bin/env bash

export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"
CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_plus_kernel.cu.o roi_plus_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py
