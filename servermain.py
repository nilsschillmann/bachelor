# -*- coding: utf-8 -*-

from pipeline import Pipeline

from skimage import io

import logging
import time
import pickle

import sys

logging.basicConfig(level=logging.INFO)


def main(argv):

    input_folder = argv[0]
    output_folder = argv[1]

    print('input folder:', input_folder)
    print('output_folder:', output_folder)
    input()






    pipe = Pipeline()



if __name__ == '__main__':
    main(sys.argv[1:])


