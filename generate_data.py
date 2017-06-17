from glob import glob
from subprocess import call
from random import random, seed
import os
import argparse
from scipy.misc import imread, imsave
from skimage.transform import resize
import numpy as np

def main(data_dir, image_size):
    
    if not os.path.exists(data_dir):
        print("data_dir {} does not exist.".format(data_dir))
        exit(1)

    for direc in [data_dir + "test/", data_dir + "train/", data_dir + "valid/"]:
        if not os.path.exists(direc):
            os.makedirs(direc)    

    # get a list of all the JPEG images in the Data dir
    input_files = glob(data_dir + "*.JPEG")

    # get indices for test, train, valid sets
    seed(42)
    split = ["test/" if random() <= 0.2 else "train/" if random() <= 0.8 else "valid/" for _ in range(0, len(input_files))]

    # for each JPEG crop the image to get a square
    cropped_files = []
    for i, fname in enumerate(input_files):
        img = imread(fname, mode='RGB')
        img = resize(img, (image_size, image_size, 3))
        imsave(data_dir + split[i] + "%s.jpg" % i, img)
        cropped_files.append("%s.jpg" % i)

    # for each cropped image run the jp2a utility to convert it to ASCII pic with width = height = 100
    for i, input_file in enumerate(cropped_files):
        with open(data_dir + split[i] + "%s.ascii" % i, 'w') as ofile:
            call(['jp2a', data_dir + split[i] + input_file, "--size={}x{}".format(image_size/2, image_size/4), "--background=light"], stdout=ofile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JPEG files in data_dir resizing to a constant size and creating ASCII art.')
    parser.add_argument('--data_dir', metavar='D', default="Data/", 
                        help="""Specify the directory where the unprocessed files can be found. 
                        The output files will be written to data_dir/test/, data_dir/train/, and data_dir/valid/. 
                        The default is 'Data/'""")
    parser.add_argument('--image_size', metavar='S', default=300, 
                        help="""Specify the dimensions of the SxS output images after resizing. 
                        The default is 300x300.""")
    args = parser.parse_args()
    main(args.data_dir, args.image_size)
    