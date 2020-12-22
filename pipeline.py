# -*- coding: utf-8 -*-
"""
@author: Nils
"""
import logging
import time
import multiprocessing as mp
from math import sqrt
from functools import wraps
from functools import lru_cache
from datetime import timedelta

import numpy as np
from skimage import color
from skimage.filters import gaussian

from ownhog import hog


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
def run(img, sigmas,  plot=False):
    '''Run the complete pipeline over a given Image.'''

    sigmas, working_sigmas = self.calculate_sigmas(
        max(img.shape)/2, self.gauß_depth)

    lab = self.convert2lab(img)
    scalespaces = self.create_scalespaces(lab, sigmas)
    differences = self.create_differences(scalespaces)
    feature_vector = self.create_feature_vector_mp(differences)

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

    pool = mp.Pool(processes=4)
    scalespaces = [pool.apply_async(self.create_scalespace,
                                    args=(img, sigmas)) for img in imgs]
    output = [p.get() for p in scalespaces]
    return output


@time_logger
def create_scalespace_chained(img, sigmas):
    '''Return a gausian scalespace for an given image.'''

    def s2(s1, s3): return sqrt(s3**2 - s1**2)
    working_sigmas = [sigmas[0]]  # sigmas i have to add up
    for s in sigmas[1:]:
        last = working_sigmas[-1]
        working_sigmas.append(s2(last, s))

    scalespace = [img]
    img_filtered = img
    for s in working_sigmas:
        img_filtered = gaussian(img_filtered, s, multichannel=False)
        scalespace.append(img_filtered)
    return scalespace


def create_scalespace(img, sigmas):
    '''Return a gausian scalespace for an given image.'''

    scalespace = [img]
    for s in sigmas:
        scalespace.append(gaussian(img, s, multichannel=False))
    return scalespace


@time_logger
def create_differences(scalespaces):
    differences = []
    for scalespace in scalespaces:
        diffs = []
        for i in range(len(scalespace)-1):
            diffs.append(scalespace[i] - scalespace[i+1])
        differences.append(diffs)
    return differences


def create_hogs_pyramid(image, depth, orientations):
    feature_vector = []
    for i in [2**a for a in range(depth)]:
        x, y = map(lambda a: a//i, image.shape[:2])
        feature_vector.append(hog(
            image,
            orientations=orientations,
            pixels_per_cell=(x, y),
            cells_per_block=(1, 1),
            multichannel=False,
            block_norm='None'))

    return np.concatenate(feature_vector)


@time_logger
def create_feature_vector_mp(differences, depth, orientations):

    diffs = np.concatenate(differences)

    # os.system("taskset -p 0xff %d" % os.getpid())
    pool = mp.Pool(processes=4)
    features = [pool.apply_async(create_hogs_pyramid,
                                 args=(diff, depth, orientations))
                for diff in diffs]
    feature_vector = [p.get() for p in features]
    return np.concatenate(feature_vector)
