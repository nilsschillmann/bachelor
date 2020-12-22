# -*- coding: utf-8 -*-
'''Starting point for the project running on the server'''


import logging
import pickle
import glob
import sys
from os import listdir
from configparser import ConfigParser
from math import sqrt
from datetime import date

from skimage import io
from skimage.transform import resize, rescale
from PIL import Image

from pipeline import Pipeline


def main():
    '''Run the pipeline for all images in a folder'''
    input_folder, \
        output_folder, \
        parameter, \
        resize_factor, \
        processes = parse_config(sys.argv[1])

    configure_logging(output_folder)

    file_names = listdir(input_folder)
    areas = []
    for name in file_names:
        img = Image.open("/".join((input_folder, name)))
        areas.append(img.size[0] * img.size[1])
    min_size = min(areas)

    for name in file_names:
        img = io.imread("/".join((input_folder, name)))
        img = resize_image(img, min_size*resize_factor)
        print(img.shape, img.shape[0]*img.shape[1])

    # sigmas, working_sigmas = self.calculate_sigmas(
    #     max(img.shape)/2, self.gauß_depth)

def resize_image(img, area):
    '''Resize an image to a specific area'''
    x, y = img.shape[:2]
    x2 = round(sqrt(area / (x/y)))
    y2 = round(sqrt(area / (y/x)))
    return resize(img, (x2, y2))


def calculate_sigmas(max_sigma, n):
    '''Calculate sigmas for gausian filters.'''

    std_max = 100
    x = std_max**(1/(n-1))
    factors = tuple((x**i / std_max) for i in range(1, n))

    sigmas = tuple(max_sigma*i for i in factors)

    return sigmas


def configure_logging(folder):
    '''configure the style and path for the logging file'''
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename=folder + '/test.log',
                                  encoding='utf-8',
                                  )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s', "%H:%M:%S"))
    root_logger.addHandler(handler)
    today = date.today()
    logging.info(f'logger set up on {today}')


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
    processes = config.getint('Options', 'processes')

    return input_folder, output_folder, parameter, resize_factor, processes


if __name__ == '__main__':
    main()
