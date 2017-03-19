#!/usr/bin/env python

import numpy as np
np.random.seed(1234)
from sklearn.datasets import load_iris
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

# Load iris dataset
dataset = load_iris()
inputs = dataset['data']
target_names = dataset['target_names']
targets = dataset['target']
targets_onehot = np_utils.to_categorical(targets, len(target_names))

# Define model
model = Sequential()
model.add(Dense(32, init='glorot_normal',
        activation='relu', input_dim=inputs.shape[1]))
model.add(Dense(len(target_names), init='glorot_uniform',
        activation='softmax'))
model.summary()

# Set loss, optimizer and evaluation metrics
model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=0.10),
        metrics=['accuracy',])

# Train
model.fit(inputs, targets_onehot, batch_size=32, nb_epoch=10)

# Validate
predictions = model.predict(inputs)
predictions_argmax = np.argmax(predictions, axis=1)
accuracy = np.sum(predictions_argmax==targets)/float(inputs.shape[0])
print('Validation accuracy: {}'.format(accuracy))

# Save model
model.save_weights('weights.h5', overwrite=True)

file_ = open('architecture.json', 'w')
file_.write(model.to_json())
file_.close()
