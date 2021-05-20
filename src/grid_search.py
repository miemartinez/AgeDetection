#!/usr/bin/env python

"""
This script uses tensorflow keras and a path to a folder directory of image data to find the optimal parameters for a convolutional neural network that can predict age from face data.
As I couldn't get tensorboard to work properly with my browser, I implemented a pyplot of the model history so one can track the development of the different models and choose the one that has highest accuracy and is least overfitting.
All outputs will be saved in a created folder called "out" located in the project directory. Outputs include summary of model architecture and model history for all possible variations of models.

For more information on the model specifications and implementation of grid search see 'grid_search_util.py' in utils folder.

Parameters:
    path: str <path-to-folder-directory>, default = "../data/face_data/face_age"
    output: str <path-to-output>, default = "../out"
    epochs: int <number-of-epochs>, default = 20
    logdir: str <path-to-log-directory>, default = "../logs"

Usage:
    grid_search.py -p <path-to-folder-directory> -o <path-to-output-folder> -e <epochs> -l <log-directory>
    
Example:
    $ python3 grid_search.py -p ../data/face_data/face_age -o ../out -e 20 -l ../logs
    
"""
# load from tensorflow library
import tensorflow as tf
import tensorboard
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

# for plotting
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# systems libraries
import argparse
import os
import sys
sys.path.append(os.path.join(".."))
from utils.grid_search_util2 import GridSearch

# argparse
ap = argparse.ArgumentParser()
# input path
ap.add_argument("-p", "--path", 
                default="../data/face_data/face_age",
                help="Path to directory containing folders with images")
# output path
ap.add_argument("-o", "--output", 
                default="../out",
                help="Path to output folder")

# number of epochs
ap.add_argument("-e", "--epochs", 
                default= 20,
                type = int,
                help="Number of epochs")

# log directory path
ap.add_argument("-l", "--logdir", 
                default="../logs",
                help="Path to log directory")
# parse
args = vars(ap.parse_args())


# defining main function
def main(args):
    '''
    Main function:
    - Make out directory
    - Create train and test data
    - Perform grid search to find best model
    '''
    # get data directory
    data_dir = args["path"]
    # get output directory
    out_folder = args["output"]
    # get number of epochs
    epochs = args["epochs"]
    
    # if it doesn't exist, create output directory
    make_out_dir(out_folder)
    
    # create train and validation data set
    train_data, val_data, img_size = create_data(data_dir)
    
    # using grid_search_utils to define, train and evaluate model and plot model history 
    grid_search = GridSearch(img_size, train_data, val_data, epochs)
                                
    

def make_out_dir(output_folder):
    '''
    Create output directory if it doesn't exist 
    '''
    # specify directory name
    dirName = os.path.join(output_folder)
    # if directory does not exist
    if not os.path.exists(dirName):
        # create directory
        os.mkdir(dirName)
        # print that it has been created
        print("Directory " , dirName ,  " Created ")
    else:   
        # print that it exists
        print("Directory " , dirName ,  " already exists")
    
    

def create_data(data_dir):
    '''
    Specify batch size, validation split size, image height and width
    Create and return train and validation data.
    '''
    # define parameters for image size
    img_size = 64
    # batch size
    batch_size = 32
    # validation split
    val_split = 0.25
    
    # create training data with tensorflow function using foldernames as labels
    train_data = image_dataset_from_directory(data_dir, # directory
                                              validation_split=val_split,
                                              subset="training",
                                              seed=42, # seed for reproducibility
                                              image_size=(img_size, img_size),
                                              batch_size=batch_size)
    
    # create validation data with tensorflow function using foldernames as labels
    val_data = image_dataset_from_directory(data_dir,
                                            validation_split=val_split,
                                            subset="validation",
                                            seed=42,
                                            image_size=(img_size, img_size),
                                            batch_size=batch_size)
    return train_data, val_data, img_size
    

# Define behaviour when called from command line
if __name__=="__main__":
    main(args = args)
    