# -*- coding: utf-8 -*-
"""
@author: Nils
"""
import logging

from functools import wraps
from functools import lru_cache

# from itertools import repeat

from math import sqrt
from skimage.transform import resize
from skimage import color
from skimage.filters import gaussian
from skimage.feature import hog

import numpy as np

import os

import time
from datetime import timedelta

import multiprocessing as mp


os.system("taskset -p 0xff %d" % os.getpid())

logging.basicConfig(level=logging.INFO)



class pipeline():
    
    def __init__(self, 
                 gauß_depth         =5, 
                 hist_orientations  =8, 
                 phog_depth         =3,
                 resize_factor      =False 
                 ):
        self.gauß_depth = gauß_depth
        self.hist_orientations  = hist_orientations
        self.phog_depth         = phog_depth
        self.resize_factor      = resize_factor
        self.sigmas = tuple([4**n for n in range(gauß_depth)])
        self.working_sigmas = None
        self.calculate_sigmas()
    
    
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
    def run(self, img, plot=False):
        
        if self.resize_factor:
            img = self.resize_image(img)
        
        lab = self.convert2lab(img)
        
        scalespaces = self.create_scalespaces(lab)
        
        differences = self.create_differences(scalespaces)
        
        feature_vector = self.create_feature_vector_mp(differences)
        
        if plot:
            import plot
            plot.plot_img(img)
            plot.plot_lab(lab)
            for scalespace in scalespaces:
                plot.plot_scalespace(scalespace, labels=[0]+list(self.sigmas))
            for diff in differences:
                plot.plot_scalespace(diff)
            parameter = {
                'gauß_depth'        :self.gauß_depth,
                'hist_orientations' :self.hist_orientations,
                'phog_depth'        :self.phog_depth,
                'resize_factor'     :self.resize_factor
                }
            plot.plot_vector(feature_vector, name='test', parameter=parameter)
        
        return feature_vector
        
        
    
    @time_logger
    def resize_image(self, img):
        x, y = (round(a*self.resize_factor) for a in img.shape[:2])
        
        return resize(img, (x, y))
    
    @time_logger
    def convert2lab(self, img):
        # convert the image in a multichannel lab image
        converted = color.rgb2lab(img)
        # split the channels
        channels = [converted[:,:,i] for i in
                    range(converted.shape[-1])]
        return channels
    
    @time_logger
    @lru_cache(maxsize=None)
    def calculate_sigmas(self):
        logging.info(f"sigmas = {self.sigmas}")
        s2 = lambda s1, s3 : sqrt(s3**2 - s1**2)
        working_sigmas = [self.sigmas[0]] # sigmas i have to add up
        for s in self.sigmas[1:]:
            last = working_sigmas[-1]
            working_sigmas.append(s2(last, s))
        
        logging.info(f"working_sigma = {working_sigmas}")
        self.working_sigmas = working_sigmas
    
    @time_logger
    def create_scalespaces(self, imgs):
        
        os.system("taskset -p 0xff %d" % os.getpid())
        pool = mp.Pool(processes=4)
        scalespaces = [pool.apply_async(self.create_scalespace,
                                        args=(img,)) for img in imgs]
        output = [p.get() for p in scalespaces]
        return output

    
    
    def create_scalespace(self, img):
        scalespace = [img]
    
        img_filtered = img
        
        for s in self.working_sigmas:
            img_filtered = gaussian(img_filtered, s, multichannel=False)
            scalespace.append(img_filtered)
        return scalespace
    
    
    # @time_logger
    # def create_scalespaces_mp(imgs, sigmas):
        
    #     os.system("taskset -p 0xff %d" % os.getpid())
    #     scalespaces = []        
    #     for img in imgs:
    #         pool = mp.Pool(processes=4)
    #         scalespace = [pool.apply_async(gaussian, 
    #                                   args=(img, s, {'multichannel':False})) for s in sigmas]
    #         output = [p.get() for p in scalespace]
    #         scalespaces.append(output)
    #     return scalespaces
        
        
    
    @time_logger
    def create_differences(self, scalespaces):
        differences = []
        for scalespace in scalespaces:
            #diffs = np.diff(np.array(scalespace), axis=0)
            diffs = []
            for i in range(len(scalespace)-1):
                diffs.append(scalespace[i] - scalespace[i+1])
            
            differences.append(diffs)
        return differences


    #@time_logger
    def create_hogs_pyramid(image, depth, orientations):
        #TODO: multiprocessing

        
        
        feature_vector = []
        for i in [2**a for a in range(depth)]:
            x, y = map(lambda a : a//i, image.shape[:2])
            feature_vector.append(hog(
                image, 
                orientations=orientations, 
                pixels_per_cell=(x,y),
                cells_per_block=(1,1), 
                multichannel=False)
            )
            
        return np.concatenate(feature_vector)
    
    @time_logger
    def create_feature_vector(differences, depth):
        diffs = np.concatenate(differences) 
        
        return np.concatenate(list(map(
            lambda a: pipeline.create_hogs_pyramid(a, depth),
            diffs)
            ))
    
    
    
    @time_logger
    def create_feature_vector_mp(self, differences):
        
        diffs = np.concatenate(differences) 
        
        #os.system("taskset -p 0xff %d" % os.getpid())
        pool = mp.Pool(processes=4)
        features = [pool.apply_async(pipeline.create_hogs_pyramid,
                                     args=(diff, self.phog_depth, self.hist_orientations)) 
                    for diff in diffs]
        feature_vector = [p.get() for p in features]
        return np.concatenate(feature_vector)
    
    
        
    