import glob
from subprocess import call

def main():
    data_dir = "Data/"

    # get a list of all the JPEG images in the Data dir
    input_files = glob.glob(data_dir + "*jpg")

    # for each JPEG run the jp2a utility to convert it to ASCII pic with width = height = 100
    [call(['jp2a', input_file, "--width=100", "--grayscale", "--output=" + data_dir + "%s.ascii" % i]) for i, input_file in enumerate(input_files)]

if __name__ == "__main__":
    main()