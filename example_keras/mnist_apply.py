#!/usr/bin/env python


# Select Theano as backend for Keras
from os import environ
environ['KERAS_BACKEND'] = 'theano'

# Set architecture of system (AVX instruction set is not supported on SWAN)
environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'


from keras.models import load_model
from keras import backend
backend.set_image_dim_ordering('tf')
import numpy as np
import sys

try:
    import png
except:
    raise Exception('Have you installed pypng with `pip install --user pypng`?')

if __name__ == '__main__':
    # Load trained keras model
    model = load_model('mnist_example.h5')

    # Get image names from arguments
    filename_images = []
    for arg in sys.argv[1:]:
        filename_images.append(arg)

    # Load images from files
    images = np.zeros((len(filename_images), 28, 28, 1))
    for i_file, file_ in enumerate(filename_images):
        pngdata = png.Reader(open(file_, 'rb')).asDirect()
        for i_row, row in enumerate(pngdata[2]):
            images[i_file, i_row, :, 0] = row

    # Predict labels for images
    labels = model.predict(images)

    numbers = np.argmax(labels, axis=1)
    print('Predict labels for images:')
    for file_, number in zip(filename_images, numbers):
        print('    {} : {}'.format(file_, number))
