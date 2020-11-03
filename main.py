# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 00:44:51 2020

@author: Nils
"""

from skimage import io

from pipeline import pipeline

#from plot import plot_vector

import logging

import time


#%%

if __name__ == '__main__':
    
    #%% load image
    # folder    = "../JenAesthetics/small/"
    # file_name = "Giovanni_Francesco_Romanelli_-_The_Finding_of_Moses_-_Google_Art_Project.jpg"
    # path = folder + file_name
    
    
    #%% load test image
    folder    = "../testpics/"
    file_name = "verlauf_links_rechts.jpg"
    #file_name = "Giovanni_Moses_test.jpg"
    
    path = folder + file_name
    
           
    #%% load file
           
    img_high = io.imread(path)
    
    #%% execute pipeline
    
    parameter = {
        'gauß_depth'        :6,
        'hist_orientations' :16,
        'phog_depth'        :3,
        'resize_factor'     :1
        }
    
    
    logging.info("\n\n")
    logging.info("#################")
    logging.info(parameter)

    pipe = pipeline(**parameter)
    
    start_time = time.time()
    
    hogs = pipe.run(img_high, plot=True)
    
    executiontime = time.time() - start_time
    
    logging.info(f'feature vecotr size: {hogs.shape[0]}')
        
    #%% plot
    
    #plot_vector(hogs, file_name, parameter, executiontime)
    
    
    #%% Testsequence
    
    # for resize_factor in [0.1, 0.2, 0.5, False]:
    #     for gauß_depth in [4, 5, 6]:
    #         for hist_orientations in [8, 16]:
    #             for phog_depth in [3, 4]:
                    
    #                 parameter = {
    #                     'gauß_depth'        :gauß_depth,
    #                     'hist_orientations' :hist_orientations,
    #                     'phog_depth'        :phog_depth,
    #                     'resize_factor'     :resize_factor
    #                     }
                    
    
    #                 logging.info("\n\n")
    #                 logging.info("#################")
    #                 logging.info(parameter)
                
    #                 pipe = pipeline(**parameter)
                    
    #                 start_time = time.time()
                    
    #                 hogs = pipe.run(img_high)
                    
    #                 executiontime = time.time() - start_time
                    
    #                 logging.info(f'feature vecotr size: {hogs.shape[0]}')
                    
    #                 #%% plot
                    
    #                 plot_vector(hogs, file_name, parameter, executiontime)
    