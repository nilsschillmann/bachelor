# -*- coding: utf-8 -*-
'''Module that contains all the functionality that do not fit elsewhere.'''

import logging
from math import sqrt
from configparser import ConfigParser
from datetime import date
from collections import namedtuple
from skimage.transform import resize


def resize_image(img, area):
    '''Resize an image to a specific area'''
    width, height = img.shape[:2]
    new_width = round(sqrt(area / (width/height)))
    new_height = round(sqrt(area / (height/width)))
    return resize(img, (new_width, new_height))


def calculate_sigmas(max_sigma, number):
    '''Calculate sigmas for gausian filters.'''

    std_max = 100
    base = std_max**(1/(number-1))
    factors = tuple((base**i / std_max) for i in range(1, number))
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
    config_parser = ConfigParser()
    config_parser.read(config_path)
    input_folder = config_parser.get('Folders', 'input')
    output_folder = config_parser.get('Folders', 'output')
    parameter = {
        'gauß_depth': config_parser.getint('Parameter', 'gauss_depth'),
        'hist_orientations': config_parser.getint(
            'Parameter', 'hist_orientations'),
        'phog_depth': config_parser.getint('Parameter', 'phog_depth'),
        }
    resize_factor = config_parser.getfloat('Options', 'resize_factor')
    processes = config_parser.getint('Options', 'processes')

    Configuration = namedtuple('Configuration', ['input_folder',
                                                 'output_folder',
                                                 'parameter',
                                                 'resize_factor',
                                                 'processes'])

    return Configuration(input_folder,
                         output_folder,
                         parameter,
                         resize_factor,
                         processes)
