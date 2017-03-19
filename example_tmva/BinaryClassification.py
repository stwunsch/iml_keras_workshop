#!/usr/bin/env python

# Select Theano as backend for Keras
from os import environ
environ['KERAS_BACKEND'] = 'theano'

# Set architecture of system (AVX instruction set is not supported on SWAN)
environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'


from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

# Setup TMVA
TMVA.PyMethodBase.PyInitialize()

output = TFile.Open('BinaryClassificationKeras.root', 'RECREATE')
factory = TMVA.Factory('TMVAClassification', output,
        '!V:!Silent:Color:DrawProgressBar:Transformations=G:AnalysisType=Classification')

# Load data
if not isfile('tmva_class_example.root'):
    call(['curl', '-O', 'http://root.cern.ch/files/tmva_class_example.root'])

data = TFile.Open('tmva_class_example.root')
signal = data.Get('TreeS')
background = data.Get('TreeB')

dataloader = TMVA.DataLoader('BinaryClassificationKeras')
for branch in signal.GetListOfBranches():
    dataloader.AddVariable(branch.GetName())

dataloader.AddSignalTree(signal, 1.0)
dataloader.AddBackgroundTree(background, 1.0)
dataloader.PrepareTrainingAndTestTree(TCut(''),
        'nTrain_Signal=4000:nTrain_Background=4000:SplitMode=Random:NormMode=NumEvents:!V')

# Generate model

# Define model
model = Sequential()
model.add(Dense(64, init='glorot_normal', activation='relu', input_dim=4))
model.add(Dense(2, init='glorot_uniform', activation='softmax'))

# Set loss and optimizer
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy',])

# Store model to file
model.save('model.h5')
model.summary()

# Book methods
factory.BookMethod(dataloader, TMVA.Types.kFisher, 'Fisher',
        '!H:!V:Fisher:VarTransform=None')
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
        'H:!V:VarTransform=G:FilenameModel=model.h5:NumEpochs=10:BatchSize=32')

# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
