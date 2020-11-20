# -*- coding: utf-8 -*-
"""
@author: Nils
"""
import logging

from functools import wraps
from functools import lru_cache

from math import sqrt
from skimage.transform import resize
from skimage import color
from skimage.filters import gaussian
# from skimage.feature import hog
from ownhog import hog

import numpy as np

import os

import time
from datetime import timedelta

import multiprocessing as mp


#os.system("taskset -p 0xff %d" % os.getpid())

logging.basicConfig(level=logging.INFO)



class Pipeline():

    def __init__(self,
                 hist_orientations  = 8,
                 gauß_depth         = 5,
                 phog_depth         = 3,
                 resize_factor      = False
                 ):

        self.hist_orientations  = hist_orientations
        self.gauß_depth         = gauß_depth
        self.phog_depth         = phog_depth
        self.resize_factor      = resize_factor



    def __repr__(self):
        return f'{self.__class__.__name__}(' \
            f'{self.hist_orientations}, ' \
            f'{self.gauß_depth}, ' \
            f'{self.phog_depth}, ' \
            f'{self.resize_factor})'


    def get_parameter(self):
        return {'hist_orientations': self.hist_orientations,
                'gauß_depth': self.gauß_depth,
                'phog_depth': self.phog_depth,
                'resize_factor': self.resize_factor}

    def time_logger(function):
        '''Decorate the given function to logg the execution time.'''

        @wraps(function)
        def wrapper(*args, **kwargs):
            logging.info(f'Execute {function.__name__} ...')
            start_time      = time.time()
            result          = function(*args, **kwargs)
            executiontime   = time.time() - start_time
            delta           = str(timedelta(seconds=executiontime))
            logging.info(f'{function.__name__:<20} executed in {delta :>20}')
            return result
        return wrapper


    @time_logger
    def run(self, img, plot=False):
        '''Run the complete pipeline over a given Image.'''

        logging.info(f'Picture shape: {img.shape}')

        if self.resize_factor and self.resize_factor != 1:
            img = self.resize_image(img)
            logging.info(f'Picture resized shape: {img.shape}')

        sigmas, working_sigmas = self.calculate_sigmas(max(img.shape)/2, self.gauß_depth)


        lab = self.convert2lab(img)
        scalespaces = self.create_scalespaces(lab, sigmas)
        differences = self.create_differences(scalespaces)
        feature_vector = self.create_feature_vector_mp(differences)

        return lab, scalespaces, differences, feature_vector


    @time_logger
    def resize_image(self, img):
        '''Return a resolution scaled version of an image.'''
        x, y = (round(a*self.resize_factor) for a in img.shape[:2])
        return resize(img, (x, y))


    @time_logger
    def convert2lab(self, img):
        '''Convert an rgb image to python list of Lab channels.'''
        converted = color.rgb2lab(img)
        # split the channels
        channels = [converted[:,:,i] for i in
                    range(converted.shape[-1])]
        return channels


    @staticmethod
    @time_logger
    @lru_cache(maxsize=None)
    def calculate_sigmas(max_sigma, n):
        '''Calculate sigmas for gausian filters.'''

        std_max = 100
        x = std_max**(1/(n-1))
        factors = tuple((x**i / std_max) for i in range(1, n))

        sigmas = tuple(max_sigma*i for i in factors)

        #sigmas             = tuple([4**n for n in range(gauß_depth)])
        #working_sigmas     = self.calculate_sigmas(self.sigmas)


        def s2(s1, s3): return sqrt(s3**2 - s1**2)
        working_sigmas = [sigmas[0]] # sigmas i have to add up
        for s in sigmas[1:]:
            last = working_sigmas[-1]
            working_sigmas.append(s2(last, s))

        sig_string = ', '.join(f'{sigma:.2f}' for sigma in sigmas)
        ws_string = ', '.join(f'{sigma:.2f}' for sigma in working_sigmas)
        logging.info(f"sigmas = " + sig_string)
        logging.info("working_sigmas = " + ws_string)

        return sigmas, working_sigmas


    @time_logger
    def create_scalespaces(self, imgs, sigmas):
        '''Return a list of gausian scalespaces for a list of images.'''
        #os.system("taskset -p 0xff %d" % os.getpid())


        pool = mp.Pool(processes=4)
        scalespaces = [pool.apply_async(self.create_scalespace,
                                        args=(img, sigmas)) for img in imgs]
        output = [p.get() for p in scalespaces]
        return output


    def create_scalespace_chained(self, img, working_sigmas):
        '''Return a gausian scalespace for an given image.'''


        scalespace = [img]
        img_filtered = img
        for s in working_sigmas:
            img_filtered = gaussian(img_filtered, s, multichannel=False)
            scalespace.append(img_filtered)
        return scalespace

    def create_scalespace(self, img, sigmas):
        '''Return a gausian scalespace for an given image.'''

        scalespace = [img]
        for s in sigmas:
            scalespace.append(gaussian(img, s, multichannel=False))
        return scalespace

    @time_logger
    def create_differences(self, scalespaces):
        differences = []
        for scalespace in scalespaces:
            diffs = []
            for i in range(len(scalespace)-1):
                diffs.append(scalespace[i] - scalespace[i+1])
            differences.append(diffs)
        return differences


    #@time_logger
    def create_hogs_pyramid(self, image):
        feature_vector = []
        for i in [2**a for a in range(self.phog_depth)]:
            x, y = map(lambda a : a//i, image.shape[:2])
            feature_vector.append(hog(
                image,
                orientations=self.hist_orientations,
                pixels_per_cell=(x,y),
                cells_per_block=(1,1),
                multichannel=False,
                block_norm = 'None')
            )

        return np.concatenate(feature_vector)


    @time_logger
    def create_feature_vector_mp(self, differences):

        diffs = np.concatenate(differences)

        #os.system("taskset -p 0xff %d" % os.getpid())
        pool = mp.Pool(processes=4)
        features = [pool.apply_async(Pipeline.create_hogs_pyramid,
                                     args=(self, diff))
                    for diff in diffs]
        feature_vector = [p.get() for p in features]
        return np.concatenate(feature_vector)


















'''

    @time_logger
    def create_feature_vector(differences, depth):
        diffs = np.concatenate(differences)

        return np.concatenate(list(map(
            lambda a: pipeline.create_hogs_pyramid(a, depth),
            diffs)
            ))


'''

