# -*- coding: utf-8 -*-

from pipeline import Pipeline

from skimage import io

import logging
import time
import pickle

import sys

from datetime import date




def main(argv):
    
    
    #### LOGING CONFIG ####
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename='../logs/test.log', 
                                  encoding='utf-8',
                                  )
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s')) # or whatever
    root_logger.addHandler(handler)

    
    logging.info(f"\n\n")
    logging.info(f" ############################# ")
    logging.info(f" #########   NEW RUN  ######## ")
    logging.info(f" ############################# ")
    logging.info(f"")
    
    
    
    
    
    # input_folder = "../JenAesthetics/test"
    # output_folder = "../features"

    # logging.info('input folder: {input_folder}')
    # logging.info('output folder: {output_folder}')
    
    

    pipe = Pipeline()
    
    

if __name__ == '__main__':
    main(sys.argv[1:])


