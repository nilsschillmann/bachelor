# -*- coding: utf-8 -*-
'''Starting point for the project running on the server'''

import logging
import pickle
import glob
import sys
from math import sqrt
from os import listdir
import multiprocessing as mp
import time
from datetime import timedelta

from skimage import io
from PIL import Image

import pipeline
import utils

Image.MAX_IMAGE_PIXELS = None


def main():
    '''Run the pipeline for all images in a folder'''
    config = utils.parse_config(sys.argv[1])
    parameter = config.parameter
    sigmas = utils.calculate_sigmas(sqrt(parameter['area'])//2,
                                    parameter['gauß_depth'])

    utils.configure_logging(config.output_folder)
    logging.info("Starting run with parameter: " + str(parameter))
    logging.info(f"sigmas: {sigmas}")

    folder_names = listdir(config.input_folder)
    file_names = []
    for folder_name in folder_names:
        file_names += listdir(config.input_folder + '/' + folder_name)
    file_map = []

    for folder_name in folder_names:

        logging.info('Start ' + folder_name)
        start_time = time.time()

        file_names = listdir("/".join([config.input_folder, folder_name]))



        executiontime = time.time() - start_time
        delta = str(timedelta(seconds=executiontime))
        logging.info(f'executed in {delta :>20}')


def run_and_save(input_path, output_path, parameter):
    img = utils.load_image(input_path)
    vector = pipeline.run(img, *parameter)
    utils.save_feature_vector(output_path, parameter, vector)

    return None


if __name__ == '__main__':
    main()
