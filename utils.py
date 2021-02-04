# -*- coding: utf-8 -*-
'''Module that contains all the functionality that do not fit elsewhere.'''

import logging
from configparser import ConfigParser
from datetime import date
from collections import namedtuple
from skimage import io
import pickle


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
        'area': config_parser.getint('Parameter', 'area'),
        }

    Configuration = namedtuple('Configuration', ['input_folder',
                                                 'output_folder',
                                                 'parameter'])

    return Configuration(input_folder,
                         output_folder,
                         parameter,
                         )


def load_image(path):
    '''Load an image file from path'''

    return io.imread(path)


def save_feature_vector(path, parameter, vector):
    '''Saves a feature vector with parameter to a file'''

    Feature_Vector = namedtuple('Feature_Vector', ('parameter', 'vector'))

    feature_vector = Feature_Vector(parameter, vector)
    with open(path, 'wb') as output:
        pickle.dump(feature_vector, output, pickle.HIGHEST_PROTOCOL)

    return None


def load_feature_vector(path):
    '''Loads and returns a feature vector with parameter from a file'''

    return pickle.load(open(path, "rb"))
