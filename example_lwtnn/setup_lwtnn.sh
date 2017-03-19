#!/bin/bash

# Get eigen
git clone https://github.com/RLovelett/eigen
cd eigen
git checkout 3.3.0
cd ..

# Set include flags for lwtnn so that it finds eigen
export LWTNN_CXX_FLAGS="-I$PWD/eigen"

# Get lwtnn
git clone https://github.com/lwtnn/lwtnn
cd lwtnn
git checkout v2.0

# Build lwtnn
make -j4
