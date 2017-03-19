#!/usr/bin/env python


# Select Theano as backend for Keras
from os import environ
environ['KERAS_BACKEND'] = 'theano'

# Set architecture of system (AVX instruction set is not supported on SWAN)
environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'


import numpy as np
np.random.seed(1234)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend
backend.set_image_dim_ordering('tf')


def download_mnist_dataset():
    '''
    Download MNIST dataset using Keras example dataloader

    Returns:
        x_train: Training images
        x_test: Testing images
        y_train: Training labels
        y_test: Testing labels
    '''

    from keras.datasets import mnist
    from keras.utils.np_utils import to_categorical

    # Download dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # The data is loaded as flat array with 784 entries (28x28),
    # we need to reshape it into an array with shape:
    # (num_images, pixels_row, pixels_column, color channels)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Convert the uint8 PNG greyscale pixel values in range [0, 255]
    # to floats in range [0, 1]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Convert digits to one-hot vectors, e.g.,
    # 2 -> [0 0 1 0 0 0 0 0 0 0]
    # 0 -> [1 0 0 0 0 0 0 0 0 0]
    # 9 -> [0 0 0 0 0 0 0 0 0 1]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    '''
    Download MINST dataset using Keras examples loader
    '''

    x_train, x_test, y_train, y_test = download_mnist_dataset()

    print('Number of train/test images: {}/{}'.format(
            x_train.shape[0], x_test.shape[0]))

    '''
    Set up the neural network architecture you want to use
    '''

    model = Sequential()

    # First hidden layer
    model.add(Convolution2D(
            4, # Number of output filter or so-called feature maps
            3, # column size of sliding window used for convolution
            3, # row size of sliding window used for convolution
            activation='relu', # Rectified linear unit
            input_shape=(28,28,1))) # 28x28 image with 1 color channel

    # All other hidden layers
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(10, activation='softmax'))

    # Print model summary
    model.summary()

    # Compile model
    model.compile(loss='categorical_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy'])

    '''
    Train
    '''

    # Set up callbacks
    checkpoint = ModelCheckpoint(
            filepath='mnist_example.h5',
            save_best_only=True)

    # Train
    model.fit(x_train, y_train, # Training data
            batch_size=100, # Batch size
            nb_epoch=10, # Number of training epochs
            validation_split=0.5, # Use 20% of the train dataset for validation
            callbacks=[checkpoint]) # Register callbacks

    '''
    Test
    '''

    # Get predictions
    y_pred = model.predict(x_test)

    # Compare predictions with test labels
    test_accuracy = np.sum(
            np.argmax(y_test, axis=1)==np.argmax(y_pred, axis=1))/float(x_test.shape[0])

    print('Test accuracy: {}'.format(test_accuracy))

    '''
    Save model
    '''

    # Save architecture and weights in one file
    # NOTE: This has to be done only if you do not use a checkpoint callback
    #model.save('mnist_example.h5')
