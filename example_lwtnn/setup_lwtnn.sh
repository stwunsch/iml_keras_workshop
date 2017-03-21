#!/bin/bash

# Get eigen
git clone https://github.com/RLovelett/eigen --depth 1
cd eigen
git checkout 3.3.0
cd ..

# Set include flags for lwtnn so that it finds eigen
export LWTNN_CXX_FLAGS="-I$PWD/eigen"

# Get lwtnn
git clone https://github.com/lwtnn/lwtnn --depth 1
cd lwtnn
git checkout v2.0

# Build lwtnn
make -j4
cd ..

# Tell your system where to find the liblwtnn.so library
export LD_LIBRARY_PATH=lwtnn/lib
