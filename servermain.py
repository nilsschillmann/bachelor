# -*- coding: utf-8 -*-
import logging
import time
import pickle
import glob
import sys
from configparser import ConfigParser

from skimage import io
from PIL import Image

from pipeline import Pipeline


class Main:

    def __init__(self, input_folder, output_folder, parameter):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.parameter = parameter

    def main(argv):
        input_folder = "../JenAesthetics/test"
        output_folder = "../features"

        file_paths = glob.glob(input_folder + "/*")

        file_sizes = []
        for file in file_paths:
            im = Image.open(file)
            file_sizes.append(im.size[0] * im.size[1])

        print(file_sizes)
        print()
        print(max(file_sizes))

        pipe = Pipeline()

        logging.info('parameter: {pipe.get_parameter()}')


def configure_logging(folder):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename = folder + '/test.log',
                                  encoding = 'utf-8',
                                  )
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    root_logger.addHandler(handler)
    logging.info('logger set up')


def load_config(file_path):
    '''load a specifig configuration from ini file'''
    config = ConfigParser()
    config.read(file_path)
    input_folder = config.get('Folders', 'input')
    output_folder = config.get('Folders', 'output')
    parameter = {
        'gauß_depth': config.getint('Parameter', 'gauss_depth'),
        'hist_orientations': config.getint('Parameter', 'hist_orientations'),
        'phog_depth': config.getint('Parameter', 'phog_depth'),
        'resize_factor': config.getfloat('Parameter', 'resize_factor'),
        }

    return input_folder, output_folder, parameter


if __name__ == '__main__':
    input_folder, output_folder, parameter = load_config(sys.argv[1:])
    configure_logging(output_folder)
    main = Main(input_folder, output_folder, parameter)
