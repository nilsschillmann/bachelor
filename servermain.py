# -*- coding: utf-8 -*-
'''Starting point for the project running on the server'''

import logging
import pickle
import glob
import sys
from math import sqrt
from os import listdir
import multiprocessing as mp

from skimage import io
from PIL import Image


import pipeline
import utils


def main():
    '''Run the pipeline for all images in a folder'''
    config = utils.parse_config(sys.argv[1])
    parameter = config.parameter

    utils.configure_logging(config.output_folder)

    file_names = listdir(config.input_folder)

    area = parameter['area']

    sigmas = utils.calculate_sigmas(sqrt(area)//2, parameter['gauß_depth'])

    logging.info(f"sigmas: {sigmas}")

    for name in file_names:
        print(name)

    pool = mp.Pool()

    procs = {name: pool.apply_async(pipeline.run, args=(config.input_folder + name,
                                                  sigmas,
                                                  parameter['phog_depth'],
                                                  parameter['hist_orientations'],
                                                  area)) for name in file_names}

    results = {name: p.get() for name, p in procs.items()}

    with open(''.join([config.output_folder, "/", 'result', ".pkl"]), 'wb') as output:
            pickle.dump((parameter, results), output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print('start')
    main()
    print('ende')
