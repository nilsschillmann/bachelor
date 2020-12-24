# -*- coding: utf-8 -*-
'''Module that contains all the functionality that do not fit elsewhere.'''

import logging
from math import sqrt
from configparser import ConfigParser
from datetime import date

from skimage.transform import resize, rescale


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
