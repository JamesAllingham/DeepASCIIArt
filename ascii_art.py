import keras
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Dropout, Input, MaxPool2D
from keras.activations import softmax
from keras.utils import to_categorical

import numpy as np
from scipy.misc import imread

from glob import glob

def ascii_to_one_hot(ascii):
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
    one_hot = to_categorical(codes, num_classes=128)

    return np.reshape(one_hot, (x, y, -1))

def one_hot_to_ascii(one_hot):
    # reshape array into 2D
    x, y, z = one_hot.shape
    reshaped = np.reshape(one_hot, (x*y, z))

    # convert from 1 hot to ascii codes
    # codes = np.sum(np.reshape(np.arange(z), (-1, z)).transpose()*reshaped.transpose(), axis=0, dtype=np.int16)
    codes = np.argmax(reshaped, axis=1)

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
    input_files = glob(data_dir + "*.jpg")[:200]
    output_files = glob(data_dir + "*.ascii")[:200]

    # read the input files as images
    input_images = [np.float32(imread(fname)) for fname in input_files]

    # convert all of the output files to 1 hot encodings
    output_ascii_strings = [open(fname, 'r').read() for fname in output_files]
    output_one_hot = np.array([ascii_to_one_hot(ascii) for ascii in output_ascii_strings])

    # only take images with size (300, 300, 3)
    temp = zip(input_images, output_one_hot)
    keep = [pair for pair in temp if pair[0].shape == (300, 300, 3)]
    input_images, output_one_hot = map(np.array, zip(*keep))

    # define network using functional API
    inputs = Input(shape=(300, 300, 3))
    x = Dropout(0.1)(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D()(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) 
    x = MaxPool2D(pool_size=(2, 1))(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) 
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x) 
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    outputs = Conv2D(128, (3, 3), activation=lambda z: softmax(z, axis=2), padding='same')(x)

    # create model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # train model
    # TO DO: write a data generator so that the input and output data can be loaded in a lazy fashion from the directory
    # TO DO: create validation set and use to evaluate the training progress
    # TO DO: add callbacks for early stopping, learning rate annealing, tensorbard 
    # TO DO: save model on completion of training
    model.fit(input_images, output_one_hot, batch_size=5, epochs=20)

    # test

    # TO DO: create test set
    # TO DO: get proper test metrics such as accuracy, loss
    test_image = np.array([np.float32(imread("test.jpg"))])
    test = model.predict(test_image)[0]
    test_ascii = one_hot_to_ascii(test)
    
    print(test_ascii)

if __name__ == "__main__":
    main()