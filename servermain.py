# -*- coding: utf-8 -*-

from pipeline import Pipeline

from skimage import io

import logging
import time
import pickle
import glob

import sys

from PIL import Image




def main(argv):
    
    
    #### LOGING CONFIG ####
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename='../logs/test.log', 
                                  encoding='utf-8',
                                  )
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    root_logger.addHandler(handler)

    
    logging.info(f"\n\n")
    logging.info(f" ############################# ")
    logging.info(f" #########   NEW RUN  ######## ")
    logging.info(f" ############################# ")
    logging.info(f"")
    
    
    
    
    
    input_folder = "../JenAesthetics/test"
    output_folder = "../features"
    
    file_paths = glob.glob(input_folder + "/*")
    
    file_sizes = []
    for file in file_paths:
        im = Image.open(file)
        file_sizes.append(im.size[0] * im.size[1])
        
    print(file_sizes)
    print()
    print(max(file_sizes))
    
    pipe = Pipeline()
    
    logging.info('parameter: {pipe.get_parameter()}')
    
    for file in file_paths:
        
    
    

if __name__ == '__main__':
    main(sys.argv[1:])


