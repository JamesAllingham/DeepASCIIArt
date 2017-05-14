from glob import glob
from subprocess import call
from scipy.misc import imread, imsave
import numpy as np

def main():
    data_dir = "Data/"

    # get a list of all the JPEG images in the Data dir
    input_files = glob(data_dir + "*.JPEG")

    # for each JPEG crop the image to get a square
    for i, fname in enumerate(input_files):
        img = np.float32(imread(fname, mode='RGB'))
        img = img[:300, :300, :]
        imsave(data_dir + "%s.jpg" % i, img)

    # for each JPEG run the jp2a utility to convert it to ASCII pic with width = height = 100
    for i, input_file in enumerate(input_files):
        with open(data_dir + "%s.ascii" % i, 'w') as ofile:
            call(['jp2a', input_file, "--size=150x150", "--background=light"], stdout=ofile)


if __name__ == "__main__":
    main()