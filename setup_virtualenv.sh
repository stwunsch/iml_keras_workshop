#!/bin/bash

# Initialize virtual Python environment
virtualenv py_virtualenv

# This activates the environment!
source py_virtualenv/bin/activate

# For Keras tutorial
pip install theano
pip install keras=="1.2.0"
pip install h5py
pip install pypng

# For TMVA tutorial
# NOTE: You need a recent installation of ROOT for this (>=v6.08)!

# For lwtnn tutorial
pip install sklearn
