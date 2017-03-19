#!/usr/bin/env python

# Select Theano as backend for Keras
from os import environ
environ['KERAS_BACKEND'] = 'theano'

# Set architecture of system (AVX instruction set is not supported on SWAN)
environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'

from ROOT import TMVA, TFile, TString
from array import array
from subprocess import call
from os.path import isfile

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
reader = TMVA.Reader("Color:!Silent")

# Load data
if not isfile('tmva_class_example.root'):
    call(['curl', '-O', 'http://root.cern.ch/files/tmva_class_example.root'])

data = TFile.Open('tmva_class_example.root')
signal = data.Get('TreeS')
background = data.Get('TreeB')

branches = {}
for branch in signal.GetListOfBranches():
    branchName = branch.GetName()
    branches[branchName] = array('f', [-999])
    reader.AddVariable(branchName, branches[branchName])
    signal.SetBranchAddress(branchName, branches[branchName])
    background.SetBranchAddress(branchName, branches[branchName])

# Book methods
reader.BookMVA('PyKeras', TString('BinaryClassificationKeras/weights/TMVAClassification_PyKeras.weights.xml'))

# Classify the training dataset and get the average response on the classes
responseSignal = 0.0
for i in range(signal.GetEntries()):
    signal.GetEntry(i)
    responseSignal += reader.EvaluateMVA('PyKeras')
responseSignal /= signal.GetEntries()

responseBackground = 0.0
for i in range(background.GetEntries()):
    background.GetEntry(i)
    responseBackground += reader.EvaluateMVA('PyKeras')
responseBackground /= background.GetEntries()

print('Average response on signal: {}'.format(responseSignal))
print('Average response on background: {}'.format(responseBackground))
