# -*- coding: utf-8 -*-
"""
@author: Nils
"""
import logging
import time
import multiprocessing as mp
from functools import wraps
from datetime import timedelta

import numpy as np
from skimage import color
from skimage.filters import gaussian

from ownhog import hog

PROCESSES = 4


def time_logger(function):
    '''Decorate the given function to logg the execution time.'''

    @wraps(function)
    def wrapper(*args, **kwargs):
        logging.info(f'Execute {function.__name__} ...')
        start_time = time.time()
        result = function(*args, **kwargs)
        executiontime = time.time() - start_time
        delta = str(timedelta(seconds=executiontime))
        logging.info(f'{function.__name__:<20} executed in {delta :>20}')
        return result
    return wrapper


@time_logger
def run(img, sigmas, depth, orientations):
    '''Run the complete pipeline over a given Image.'''

    lab = convert2lab(img)
    scalespaces = create_scalespaces(lab, sigmas)
    differences = create_differences(scalespaces)
    feature_vector = create_feature_vector_mp(differences, depth, orientations)

    return lab, scalespaces, differences, feature_vector


@time_logger
def convert2lab(img):
    '''Convert an rgb image to python list of Lab channels.'''
    converted = color.rgb2lab(img)
    # split the channels
    channels = [converted[:, :, i] for i in
                range(converted.shape[-1])]
    return channels


@time_logger
def create_scalespaces(imgs, sigmas):
    '''Return a list of gausian scalespaces for a list of images.'''
    # os.system("taskset -p 0xff %d" % os.getpid())

    pool = mp.Pool(processes=PROCESSES)
    scalespaces = [pool.apply_async(create_scalespace,
                                    args=(img, sigmas)) for img in imgs]
    output = [p.get() for p in scalespaces]
    return output


def create_scalespace(img, sigmas):
    '''Return a gausian scalespace for an given image.'''

    scalespace = [img]
    for sigma in sigmas:
        scalespace.append(gaussian(img, sigma, multichannel=False))
    return scalespace


@time_logger
def create_differences(scalespaces):
    '''Return the differences of images in a scalespace'''

    differences = []
    for scalespace in scalespaces:
        diffs = []
        for i in range(len(scalespace)-1):
            diffs.append(scalespace[i] - scalespace[i+1])
        differences.append(diffs)
    return differences


def create_hogs_pyramid(image, depth, orientations):
    '''Return a vector of an histogram of oriented gradience pyramid'''

    feature_vector = []
    for i in [2**a for a in range(depth)]:
        width, height = tuple(a//i for a in image.shape[:2])
        feature_vector.append(hog(
            image,
            orientations=orientations,
            pixels_per_cell=(width, height),
            cells_per_block=(1, 1),
            multichannel=False,
            block_norm='None'))

    return np.concatenate(feature_vector)


@time_logger
def create_feature_vector_mp(differences, depth, orientations):
    '''Create the full feature vector over all differences images'''

    diffs = np.concatenate(differences)

    # os.system("taskset -p 0xff %d" % os.getpid())
    pool = mp.Pool(processes=PROCESSES)
    features = [pool.apply_async(create_hogs_pyramid,
                                 args=(diff, depth, orientations))
                for diff in diffs]
    feature_vector = [p.get() for p in features]
    return np.concatenate(feature_vector)
