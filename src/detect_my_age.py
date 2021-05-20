
#!/usr/bin/env python

"""
This script trains a convolutional neural network on face data to classify these into ages. Similarly, the script takes an image and a region of interest (containing a face in the image). It then crops and saves the image of the ROI and uses this for model prediction. 
In the Github repo there is a folder in the data folder called test_data. In this, I have included two images. One of myself called IMG_3864.JPG and one of my family called DSC04818.JPG. These can be used to test the model.

I have predefined the bounding boxes for these test images.

If using the provided test data:
    - IMG_3864.JPG - default:
        Face - age 21:
            x_start = 1450
            y_start = 600
            x_end = 1900
            y_end = 1050
        
    - DSC04818.JPG:
        First face - age 58:
            x_start = 615
            y_start = 400
            x_end = 745
            y_end = 530
        
        Second face - age 24:
            x_start = 790
            y_start = 390
            x_end = 910
            y_end = 510
        
        Third face - age 56:
            x_start = 1060
            y_start = 340
            x_end = 1170
            y_end = 450

Parameters:
    path2folder: str <path-to-folder-directory>, default = "../data/face_data/face_age"
    output: str <path-to-output>, default = "../out/age_detection"
    epochs: int <number-of-epochs>, default = 20
    path2image: str <path-to-unseen-image>, default = "../data/test_data/IMG_3864.JPG"
    x_start: int <start-of-ROI-on-x-axis>, default = 1450
    x_end: int <end-of-ROI-on-x-axis>, default = 1900
    y_start: int <start-of-ROI-on-y-axis>, default = 600
    y_end: int <end-of-ROI-on-y-axis>, default = 1050

Usage:
    detect_my_age.py -p <path-to-folder> -o <path-to-output> -e <epochs> -i <path-to-image>, -x_s <start-on-x-axis> -x_e <end-on-x-axis> -y_s <start-on-y-axis> -y_e <end-on-y-axis>
    
Example:
    $ python3 detect_my_age.py -p ../data/face_data/face_age -o ../out/age_detection -e 20 -i ../data/test_data/IMG_3864.jpg -x_s 1450 -x_e 1900 -y_s 600 -y_e 1050
    
## Task
- Defining model architecture, train and test model and return model
- Taking an unseen image, crop image to only contain face by using user specified region of interest, save cropped image.
- Feed path to the saved cropped image to model to get age prediction.
"""
# data and systems libraries
import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import cv2
sys.path.append(os.path.join(".."))
from contextlib import redirect_stdout

# plot libraries 
import seaborn as sns
import matplotlib.pyplot as plt

# load from tensorflow library
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
# load from sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



# argparse
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--path2folder", 
                default="../data/face_data/face_age",
                help="Path to directory containing folders with images")

ap.add_argument("-o", "--output", 
                default="../out/age_detection",
                help="Path to output folder")

ap.add_argument("-e", "--epochs", 
                default= 20,
                type = int,
                help="Number of epochs")

# adding arguments for input and output filepath
ap.add_argument("-i", "--path2image", 
                default = "../data/test_data/IMG_3864.JPG", 
                help= "Filepath to image")

# adding arguments to define region of interest
ap.add_argument("-x_s", "--x_start",
                default = 1450,
                type = int,
                help = "Start of region of interest on the x-axis")

ap.add_argument("-x_e", "--x_end", 
                default = 1900,
                type = int,
                help = "End of region of interest on the x-axis")

ap.add_argument("-y_s", "--y_start",
                default = 600,
                type = int,
                help = "Start of region of interest on the y-axis")

ap.add_argument("-y_e", "--y_end",
                default = 1050,
                type = int,
                help = "End of region of interest on the y-axis")

# parsing arguments
args = vars(ap.parse_args())


def main(args):
    '''
    Main function:
    1. Define, train and test model. Save model architecture, model history and classification report.
    2. Crop unseen image, test model on image and save prediction probabilies
    '''
    # define path to model
    folder_path = args["path2folder"]
    # epochs
    epochs = args["epochs"]
    
    output = args["output"]
    create_out_dir(output)
    
    # print in terminal
    print('\n[INFO] --- Processing data')
    # create train and test data
    X_train, X_test, y_train, y_test, img_size, classes = create_data(folder_path)
    
    # define hyperparameters
    num_units = 256
    dropout = 0.2
    optimizer = 'adam'
    batch_size = 32
    
    # define and compile model
    model = define_model(y_train, img_size, classes, num_units, dropout, optimizer)
    
    # print in terminal
    print('\n[INFO] --- Training model')
    print(f'Num_units: {num_units}, Dropout rate : {dropout}, Optimizer : {optimizer}')
    # train model on training data
    H = train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size)
    
    # plot model history
    plot_history(H, epochs, num_units, dropout, optimizer)
    
    evaluate_model(model, classes, X_test, y_test, num_units, dropout, optimizer)
    
    
    print('\n[INFO] --- Processing image')
    # define path to image
    image_path = args["path2image"]
    # load image
    image = cv2.imread(image_path)
    
    # define region of interest for cropping
    x_start = args["x_start"]
    y_start = args["y_start"]
    x_end = args["x_end"]
    y_end = args["y_end"]
    
    cropped_image, cropped_path = crop_image(image, x_start, y_start, x_end, y_end)
    
    # testing the model on the cropped image
    test_model(cropped_path, model, img_size, classes)

    

def create_out_dir(output_folder):
    '''
    Create out directory if it doesn't exist in the data folder
    '''
    dirName = os.path.join(output_folder)
    if not os.path.exists(dirName):
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
    # empty holders for image arrays and labels
    X = []
    y = []
    # defining image size for resizing
    img_size = 64 
    
    # for the folder name and image in image folder directory
    for folder_name,_,filenames in os.walk(data_dir):
        # if the folder name is not face_age nor data
        if folder_name !="face_age" and folder_name != 'data':
            # take each file
            for file in filenames:
                # define file path
                file_path = folder_name +"/"+ file
                # open image from file path
                image = Image.open(file_path)
                # convert image to color image
                image = image.convert('RGB')
                # resize to 64x64
                image = image.resize((img_size, img_size))
                # append image as array to X
                X.append(np.array(image))
                # append folder name to y
                y.append(folder_name[-3:])
        # if folder name is face_age or data
        else:
            pass
    
    # convert from list to array
    X = np.array(X)
    y = np.array(y)
    
    # split data to train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25, 
                                                        random_state=42,
                                                        stratify=y)
    
    # Min-Max scaling
    X_train = (X_train - X_train.min())/(X_train.max() - X_train.min())
    X_test = (X_test - X_test.min())/(X_test.max() - X_test.min())
    
    # define classes
    classes = sorted(set(y_train))
    
    # convert labels from integers to vectors (one hot encoding)
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)

    return X_train, X_test, y_train, y_test, img_size, classes



def define_model(y_train, img_size, classes, num_units, dropout, optimizer):
    '''
    Defining the model architecture and saving this as both a txt and png file in the out folder.
    Returning the model to be used globally.
    '''
    num_classes = len(classes)
    
    model = tf.keras.models.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_size, img_size, 3)), # add random horizontal flip
        layers.experimental.preprocessing.RandomRotation(0.1), # add random rotation
        layers.Conv2D(16, 3, padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(num_units, activation=tf.nn.relu),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation = 'softmax')
        ])

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  )
    
    # Model summary
    model_summary = model.summary()
    
    # name for saving model summary
    model_name = f"model_summary_{num_units}_{dropout}_{optimizer}.txt" 
    model_path = os.path.join("..", "out", "age_detection", model_name)
    # Save model summary
    with open(model_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    
    # name for saving plot
    plot_name = f"model_summary_{num_units}_{dropout}_{optimizer}.png" 
    plot_path = os.path.join("..", "out", "age_detection", plot_name)
    # Visualization of model
    model_plot = tf.keras.utils.plot_model(model,
                            to_file = plot_path,
                            show_shapes=True,
                            show_layer_names=True)
    
    print(f"\n[INFO] Model architecture is saved as txt in '{model_path}' and as png in '{plot_path}'.")
    
    return model
    
    

def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    """
    Training the model on the training data and validating it on the validation data.
    """
    # Train model
    H = model.fit(X_train, y_train, 
                  validation_data = (X_test, y_test),
                  epochs=epochs, 
                  verbose=1) 

    return H
    
    
def plot_history(H, epochs, num_units, dropout, optimizer):
    """
    Plotting the loss/accuracy of the model during training and saving this as a png file in the out folder.
    """
    # name for saving output
    figure_name = f"model_history_{num_units}_{dropout}_{optimizer}.png"
    figure_path = os.path.join("..", "out", "age_detection", figure_name)
    # Visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    
    print(f"\n[INFO] Loss and accuracy across on training and validation is saved as '{figure_path}'.")
    
def evaluate_model(model, classes, X_test, y_test, num_units, dropout, optimizer):
    """
    Evaluating the trained model and saving the classification report in the out folder. 
    """    
    
    # Predictions
    y_pred = model.predict(X_test)
    
    
    # Classification report
    classification = classification_report(y_test.argmax(axis=1), 
                                           y_pred.argmax(axis=1))
            
    # Print classification report
    print(classification)
    
    # name for saving report
    report_name = f"classification_report_{num_units}_{dropout}_{optimizer}.txt"
    report_path = os.path.join("..", "out", "age_detection", report_name)
    
    # Save classification report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(classification_report(y_test.argmax(axis=1),
                                           y_pred.argmax(axis=1)))
    
    print(f"\n[INFO] Classification report is saved as '{report_path}'.")

def test_model(path2image, model, img_size, classes):
    '''
    Testing the neural network model on an unseen image.
    Saving the model probabilities for the different digits.
    Printing the label with the highest probability.
    '''
    print("\n--Testing the image you gave me--")
    # opening image from path
    test_image = cv2.imread(path2image)
    # compressing image to match the classifier
    test_image = cv2.resize(test_image, (64, 64), interpolation=cv2.INTER_AREA)
    # min max scalling
    test_image = (test_image - test_image.min())/(test_image.max() - test_image.min())
    
    # Reshape array
    #test_probs = model.predict(test_image)
    test_probs = model.predict(test_image.reshape(1,64,64,3))
    # plot prediction
    plot_path = os.path.join("..", "out", "age_detection", "label_predictions.png")
    
    plt.figure()
    sns_plot = sns.barplot(x=classes, 
                            y=test_probs.squeeze());
    plt.ylabel("Probability");
    plt.xlabel("Class")
    fig = sns_plot.get_figure()
    fig.savefig(plot_path)
    # print that the figure has been saved
    print(f"\nThe label probability plot is saved as {plot_path}")
     
    # find and save the label with highest probability
    idx_cls = np.argmax(test_probs)
    # print predictied label
    print(f"I think that the person in the image is {classes[idx_cls]} years old")
       
def crop_image(image, x_start, y_start, x_end, y_end):
    '''
    Cropping the original image to the extent of the ROI.
    Saving this in the output folder as jpg.
    Returns the cropped image.
    '''
    # cropping image using numpy slicing and the specified x and y coordinates
    image_cropped = image[y_start:y_end, x_start: x_end]
    # save cropped image as jpg
    cropped_path = os.path.join("..", "data", "test_data", "image_cropped.jpg")
    # make output directory if it does not exist
    crop_dir = os.path.join("..", "data", "test_data")
    create_out_dir(crop_dir)
    # save image
    cv2.imwrite(cropped_path, image_cropped)
    # print that file has been saved
    print(f"\nThe cropped image is saved as {cropped_path}")
      
    return image_cropped, cropped_path
      
if __name__ == "__main__":
    main(args)

    