# -*- coding: utf-8 -*-
'''Starting point for the project running on the server'''

import logging
import pickle
import glob
import sys
from math import sqrt
from os import listdir

from skimage import io
from PIL import Image

import pipeline
import utils


def main():
    '''Run the pipeline for all images in a folder'''
    config = utils.parse_config(sys.argv[1])
    parameter = config.parameter

    utils.configure_logging(config.output_folder)

    file_names = listdir(config.input_folder)

    # checkout the smalest picture by area
    areas = []
    for name in file_names:
        img = Image.open("/".join((config.input_folder, name)))
        areas.append(img.size[0] * img.size[1])
    min_size = min(areas)

    sigmas = utils.calculate_sigmas(sqrt(min_size)//2, parameter.gauß_depth)

    logging.info(f"sigmas: {sigmas}")

    for name in file_names:
        img = io.imread("/".join((config.input_folder, name)))
        img = utils.resize_image(img, min_size*config.resize_factor)
        print(img.shape, img.shape[0]*img.shape[1])


if __name__ == '__main__':
    main()
