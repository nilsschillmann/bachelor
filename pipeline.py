# -*- coding: utf-8 -*-
"""
@author: Nils
"""
import logging

from functools import wraps

from math import sqrt
from skimage.transform import resize
from skimage import color
from skimage.filters import gaussian
from skimage.feature import hog

import numpy as np

import time
from datetime import timedelta

#from multiprocessing import Pool


logging.basicConfig(level=logging.INFO)



class pipeline():
    
    def __init__(self, resize_=False, sigmas=[4**n for n in range(5)]):
        self.resize_factor = resize_
        self.sigmas = sigmas
    
    
    def time_logger(function):
        """
        Decorate the given function to logg the execution time.

        Parameters
        ----------
        function : function

        Returns
        -------
        wrapper function

        """

        @wraps(function)
        def wrapper(*args, **kwargs):
            logging.info(f'execute {function.__name__}')
            start_time = time.time()
            result = function(*args, **kwargs)
            executiontime = time.time() - start_time
            logging.info('{} executed in {}'.
                         format(function.__name__, timedelta(seconds=executiontime)))
            return result
        return wrapper
    
    
    @time_logger
    def run(self, img):
        
        if self.resize_factor:
            img = pipeline.resize_image(img, self.resize_factor)
        
        lab = pipeline.convert2lab(img)
        
        working_sigmas = pipeline.calculate_sigmas(self.sigmas)
        
        scalespaces = pipeline.create_scalespaces(lab, working_sigmas)
        
        differences = pipeline.create_differences(scalespaces)
        
        #hogs = self.create_hogs(differences)
        feature_vector = self.create_feature_vector(differences, 3)
        
        return feature_vector
        
        
    
    @time_logger
    def resize_image(img, factor):
        x, y = map(lambda a: round(a*factor), img.shape[:2])
        
        return resize(img, (x, y))
    
    @time_logger
    def convert2lab(img):
        # convert the image in a multichannel lab image
        converted = color.rgb2lab(img)
        # split the channels
        channels = [converted[:,:,i] for i in
                    range(converted.shape[-1])]
        return channels
    
    @time_logger
    def calculate_sigmas(sigmas):
        logging.info(f"sigmas = {sigmas}")
        s2 = lambda s1, s3 : sqrt(s3**2 - s1**2)
        working_sigmas = [sigmas[0]] # sigmas i have to add up
        for s in sigmas[1:]:
            last = working_sigmas[-1]
            working_sigmas.append(s2(last, s))
        
        logging.info(f"working_sigma = {working_sigmas}")
        return working_sigmas
    
    @time_logger
    def create_scalespaces(imgs, working_sigmas):
        scalespaces = []
        for img in imgs:
            scalespace = [img]
    
            img_filtered = img
            
            for s in working_sigmas:
                img_filtered = gaussian(img_filtered, s, multichannel=False)
                scalespace.append(img_filtered)
            scalespaces.append(scalespace)
        return scalespaces
    
    @time_logger
    def create_differences(scalespaces):
        differences = []
        for scalespace in scalespaces:
            diffs = np.diff(np.array(scalespace), n=1, axis=0)
            
            differences.append(diffs)
        return np.concatenate(differences) 


    #@time_logger
    def create_hogs_pyramid(image, depth):
        
        feature_vector = []
        for i in [4**a for a in range(depth)]:
            x, y = map(lambda a : a//i,image.shape[:2])
            feature_vector.append(hog(image, orientations=8, pixels_per_cell=(x,y),cells_per_block=(1,1), multichannel=False))
            
        return np.concatenate(feature_vector)
    
    @time_logger
    def create_feature_vector(self, differences, depth):
        return np.concatenate(list(map(lambda a: pipeline.create_hogs_pyramid(a, depth), differences)))
        
        # return list(map(lambda image: hog(
        #         image, 
        #         orientations=8, 
        #         pixels_per_cell=(16, 16),
        #         cells_per_block=(1, 1), 
        #         visualize=False, 
        #         multichannel=False,
        #         feature_vector=True
        #     ), images))
    

    

    

    