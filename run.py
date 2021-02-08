#!/usr/bin/env python

import argparse
import os
import sys
import json
import shutil

from src.gradcam import *

data_ingest_params = './config/data-params.json'
fp_params = './config/file_path.json'
gradcam_params = './config/gradcam_params.json'
train_params = './config/train_params.json'

def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)
    return param

def main(targets):
    
    if 'clean' in targets:
        shutil.rmtree('results/', ignore_errors=True)
     
    if 'train' in targets:
        if not os.path.isdir('results'):
            os.makedirs('results') 
        train_fp = load_params(fp_params)['train_path']
        input_train_params = load_params(train_params)
        batch_size = input_train_params['batch_size']
        learning_rate = input_train_params['learning_rate']
        num_epochs = input_train_params['num_epochs']
        os.system("python " + train_fp + " --batch-size " + str(batch_size) + " --learning-rate " + str(learning_rate) + " --num-epochs " + str(num_epochs) + " --use-cuda")
        
    # This is currently executing the gradcam.py file (as specified in notebook_path). This needs to be changed! (Should generate performance table instead)
    if "test" in targets:       
        if not os.path.isdir('results'):
            os.makedirs('results')            
        # This is "python gradcam.py --image-path test/COCOimg.jpg" (Needs to be changed as well)
        os.system("python " + load_params(data_ingest_params)["notebook_path"] + " --image-path "+ load_params(data_ingest_params)["img"])
        
    if "gradcam" in targets:      
        # Check if directory "results" is created
        if not os.path.isdir('results'):
            os.makedirs('results')
        gradcam_fp = load_params(fp_params)['gradcam_path']
        input_gradcam_params = load_params(gradcam_params)
        input_images = input_gradcam_params["load_image_path"]["image_input_path_train_covered"]
        save_images = input_gradcam_params['save_image_path']
        #print("python " + gradcam_fp + " --image-path " + input_images + " --use_cuda " + input_cuda)
        os.system("python " + gradcam_fp + " --image-path " + input_images + " --save-path-gb " + save_images['gb_path'] + " --save-path-cam-gb " + save_images['cam_gb_path'] + " --save-path-cam " + save_images['cam_path'] + " --use-cuda")
        

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)

