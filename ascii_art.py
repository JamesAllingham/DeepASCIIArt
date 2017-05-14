import keras
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Dropout, Input, Flatten, Dense, Activation, MaxPool2D
from keras.activations import softmax
from keras.utils import to_categorical

import numpy as np
from scipy.misc import imread

from glob import glob

def ascii_to_1_hot(ascii):
    # split ascii string by row
    rows = ascii.splitlines()

    # for each row, get a list of chars
    chars = [list(row) for row in rows]

    # for each char, replace it with the int ASCII code
    codes = np.array([[ord(chr) for chr in chrs] for chrs in chars])

    # flatten temporarily
    x, y = codes.shape
    codes = np.hstack(codes)

    # convert to 1 hot
    one_hot =  to_categorical(codes, num_classes=128)

    return np.reshape(one_hot, (x, y, -1))

def one_hot_to_ascii(one_hot):
    # reshape array into 2D
    x, y, z = one_hot.shape
    reshaped = np.reshape(one_hot, (x*y, z))

    # convert from 1 hot to ascii codes
    codes = np.sum(np.reshape(np.arange(z), (-1, z)).transpose()*reshaped.transpose(), axis=0, dtype=np.int16)

    # convert back to correct shape
    codes = np.reshape(codes, (x, y))

    # convert to a list of list of chars
    chars = [[chr(num) for num in nums] for nums in codes]

    # convert to a list of strings
    strs = ["".join(row) for row in chars]

    # join rows by newline
    return "\n".join(strs)


def main():
    data_dir = "Data/"

    # get the paths for all of the input and output files
    input_files = glob(data_dir + "*.jpg")[:100]
    output_files = glob(data_dir + "*.ascii")[:100]

    # read the input files as images
    input_images = [np.float32(imread(fname)) for fname in input_files]

    # convert all of the output files to 1 hot encodings
    output_ascii_strings = [open(fname, 'r').read() for fname in output_files]
    output_1_hot = np.array([ascii_to_1_hot(ascii) for ascii in output_ascii_strings])

    # only take images with size (300, 300, 3)
    temp = zip(input_images, output_1_hot)
    keep = [pair for pair in temp if pair[0].shape == (300, 300, 3)]
    input_images, output_1_hot = map(np.array, zip(*keep))

    # define network using functional API
    inputs = Input(shape=(300, 300, 3))
    x = BatchNormalization()(inputs)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D()(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) 
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) 
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    outputs = Conv2D(128, (3, 3), activation=lambda z: softmax(z, axis=2), padding='same')(x)

    # create model
    model = Model(input=inputs, output=outputs)
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    # train model
    model.fit(input_images, output_1_hot, batch_size=5, epochs=10)

if __name__ == "__main__":
    main()