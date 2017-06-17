# Train a neural network to convert JPG images to ASCII
# Author: James Allingham
# Date: 12/05/17

from itertools import cycle, zip_longest
from glob import glob
import os
import argparse

import keras
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Dropout, Input, MaxPool2D
from keras.activations import softmax
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np
from scipy.misc import imread

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

def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def generate_batches_from_directory(path, batch_size=10):
    # get the paths for all of the input and output files
    input_files = glob(path + "*.jpg")
    output_files = glob(path + "*.ascii")
    file_pairs = zip(input_files, output_files)
    file_pairs_infinite = cycle(file_pairs)
    file_pairs_grouped = grouper(batch_size, file_pairs_infinite)
    
    # loop infinitely    
    while 1:        
        input_files, output_files = zip(*next(file_pairs_grouped)) 

        # read the input files as images
        input_images = [np.float32(imread(fname)) for fname in input_files]

        # convert all of the output files to 1 hot encodings
        output_ascii_strings = [open(fname, 'r').read() for fname in output_files]
        output_one_hot = np.array([ascii_to_one_hot(ascii) for ascii in output_ascii_strings])

        # only take images with size (300, 300, 3)
        temp = zip(input_images, output_one_hot)
        keep = [pair for pair in temp if pair[0].shape == (300, 300, 3)]

        # if there isn't anything to return then go to the next iteration
        if len(keep) == 0: continue

        # convert to numpy arrays
        input_images, output_one_hot = map(np.array, zip(*keep))

        yield input_images, output_one_hot


def main(data_dir, batch_size, image_size):
    if not os.path.exists(data_dir):
        print("data_dir {} does not exist.".format(data_dir))
        exit(1)

    # define network using functional API
    inputs = Input(shape=(image_size, image_size, 3))
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
    model.fit_generator(generate_batches_from_directory(data_dir + "train/", batch_size=batch_size), steps_per_epoch=100, epochs=100, 
                        validation_data=generate_batches_from_directory(data_dir + "valid/", batch_size=batch_size), validation_steps=20,
                        callbacks=[
                            TensorBoard(log_dir='./logs'),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto', epsilon=1e-4, cooldown=5, min_lr=0),
                            ModelCheckpoint(filepath="Models/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', save_best_only=True),
                            EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
                        ])

    # test
    # TO DO: get proper test metrics such as accuracy, loss
    test_image = np.array([np.float32(imread("test.jpg"))])
    test = model.predict(test_image)[0]
    test_ascii = one_hot_to_ascii(test)
    
    print(test_ascii)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains a model to convert JPG images to ASCII art.')
    parser.add_argument('--batch_size', metavar='B', default=5, 
                        help="""Specify the batch size. 
                        The default is 5""")
    parser.add_argument('--data_dir', metavar='D', default="Data/", 
                        help="""Specify the directory where the input and output files can be found. 
                        The directory should contain data_dir/test/, data_dir/train/, and data_dir/valid/. 
                        The default is 'Data/'""")
    parser.add_argument('--image_size', metavar='S', default=300, 
                        help="""Specify the dimensions of the SxS input images. 
                        The default is 300x300.""")
    args = parser.parse_args()
    main(args.data_dir, args.batch_size, args.image_size)