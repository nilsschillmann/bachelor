# -*- coding: utf-8 -*-
'''Starting point for the project running on the server'''


import logging
import pickle
import glob
import sys
from configparser import ConfigParser

from skimage import io
from PIL import Image

from pipeline import Pipeline


def main():
    '''Run the pipeline for all images in a folder'''
    input_folder, \
        output_folder, \
        parameter = parse_config(sys.argv[1])

    configure_logging(output_folder)

    picture_paths = glob.glob(input_folder + "/*")
    file_sizes = []
    for file in picture_paths:
        img = Image.open(file)
        file_sizes.append(img.size[0] * img.size[1])

    print(file_sizes)
    print()
    print(max(file_sizes))
    print(min(file_sizes))


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
        'resize_factor': config.getfloat('Parameter', 'resize_factor'),
        }

    return input_folder, output_folder, parameter


if __name__ == '__main__':
    main()
