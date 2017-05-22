from glob import glob
from subprocess import call
from scipy.misc import imread, imsave
import numpy as np
from random import random, seed

def main():
    data_dir = "Data/"

    # get a list of all the JPEG images in the Data dir
    input_files = glob(data_dir + "*.JPEG")

    # get indices for test, train, valid sets
    seed(42)
    split = ["test/" if random() <= 0.2 else "train/" if random() <= 0.8 else "valid/" for _ in range(0, len(input_files))]

    # for each JPEG crop the image to get a square
    for i, fname in enumerate(input_files):
        img = np.float32(imread(fname, mode='RGB'))
        img = img[:300, :300, :]
        imsave(data_dir + split[i] + "%s.jpg" % i, img)

    # for each JPEG run the jp2a utility to convert it to ASCII pic with width = height = 100
    for i, input_file in enumerate(input_files):
        with open(data_dir + split[i] + "%s.ascii" % i, 'w') as ofile:
            call(['jp2a', input_file, "--size=150x75", "--background=light"], stdout=ofile)


if __name__ == "__main__":
    # TO DO: make the source destination for the data a commandline arg
    main()