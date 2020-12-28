# -*- coding: utf-8 -*-
'''Starting point for the project running on the server'''


import pickle
import glob
import sys
from os import listdir

from skimage import io
from PIL import Image

import pipeline
import utils


def main():
    '''Run the pipeline for all images in a folder'''
    config = utils.parse_config(sys.argv[1])

    utils.configure_logging(config.output_folder)

    file_names = listdir(config.input_folder)
    areas = []
    for name in file_names:
        img = Image.open("/".join((config.input_folder, name)))
        areas.append(img.size[0] * img.size[1])
    min_size = min(areas)

    for name in file_names:
        img = io.imread("/".join((config.input_folder, name)))
        img = utils.resize_image(img, min_size*config.resize_factor)
        print(img.shape, img.shape[0]*img.shape[1])

    # sigmas, working_sigmas = self.calculate_sigmas(
    #     max(img.shape)/2, self.gauß_depth)


if __name__ == '__main__':
    main()
