# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 00:44:51 2020

@author: Nils
"""

from skimage import io

from pipeline import pipeline

import logging

#%% load image
path = r"../JenAesthetics/small/Giovanni_Francesco_Romanelli_-_" \
       "The_Finding_of_Moses_-_Google_Art_Project.jpg"
       
img_high = io.imread(path)

#%% execute pipeline
#pipe = pipeline(resize_=0.5)
pipe = pipeline(resize_=False)
hogs = pipe.run(img_high)

logging.info(f'feature vecotr size: {hogs.shape[0]}')