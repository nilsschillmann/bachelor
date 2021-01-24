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

    results = {}
    for folder_name in folder_names:

        logging.info('Start ' + folder_name)
        start_time = time.time()

        file_names = listdir("/".join([config.input_folder, folder_name]))

        logging.info('Number of Files: ' + str(len(file_names)))

        folder_path = '/'.join([config.input_folder, folder_name])

        pool = mp.Pool()
        procs = {name: pool.apply_async(pipeline.run, args=('/'.join([folder_path, name]),
                                                      sigmas,
                                                      parameter['phog_depth'],
                                                      parameter['hist_orientations'],
                                                      parameter['area']))
                  for name in file_names}

        results[folder_name] = {name: p.get() for name, p in procs.items()}

        executiontime = time.time() - start_time
        delta = str(timedelta(seconds=executiontime))
        logging.info(f'executed in {delta :>20}')

    output_file_name = '_'.join([str(p)[:3] + str(v) for p, v in parameter.items()]) + '.pkl'
    with open('/'.join([config.output_folder, output_file_name]), 'wb') as output:
            pickle.dump((parameter, results), output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
