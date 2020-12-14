# -*- coding: utf-8 -*-
'''Starting point for the project running on the server'''


import logging
import pickle
import glob
import sys
from os import listdir
from configparser import ConfigParser
from math import sqrt

from skimage import io
from skimage.transform import resize, rescale
from PIL import Image

from pipeline import Pipeline


def main():
    '''Run the pipeline for all images in a folder'''
    input_folder, \
        output_folder, \
        parameter, \
        resize_factor = parse_config(sys.argv[1])

    configure_logging(output_folder)

    picture_paths = glob.glob(input_folder + "/*")
    file_sizes = {}
    for file in listdir(input_folder):
        img = Image.open("/".join((input_folder, file)))
        file_sizes[file] = img.size[0] * img.size[1]

    min_size = min(file_sizes.values())
    # for key, value in file_sizes.items():
    #     file_sizes[key] = (min_size/value) * resize_factor

    for img_name, size in file_sizes.items():
        print(size)
        img = io.imread("/".join((input_folder, img_name)))
        img = resize_image(img, min_size*resize_factor)
        print(img.shape, img.shape[0]*img.shape[1])


def resize_image(img, area):
    '''Resize an image to a specific area'''
    x, y = img.shape[:2]
    x2 = round(sqrt(area / (x/y)))
    y2 = round(sqrt(area / (y/x)))
    return resize(img, (x2, y2))




def configure_logging(folder):
    '''configure the style and path for the logging file'''
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename=folder + '/test.log',
                                  encoding='utf-8',
                                  )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s'))
    root_logger.addHandler(handler)
    logging.info('logger set up')


def parse_config(config_path):
    '''load a specifig configuration from ini file'''
    config = ConfigParser()
    config.read(config_path)
    input_folder = config.get('Folders', 'input')
    output_folder = config.get('Folders', 'output')
    parameter = {
        'gauß_depth': config.getint('Parameter', 'gauss_depth'),
        'hist_orientations': config.getint(
            'Parameter', 'hist_orientations'),
        'phog_depth': config.getint('Parameter', 'phog_depth'),
        }
    resize_factor = config.getfloat('Options', 'resize_factor')

    return input_folder, output_folder, parameter, resize_factor


if __name__ == '__main__':
    main()
