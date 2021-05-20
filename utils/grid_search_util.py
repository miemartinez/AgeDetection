#!/usr/bin/env python
"""
Utility script for hyperparameter tuning.

Makes a class object that holds training and validation data, image size and number of epochs.

Define model architecture, train and evaluate several models with changing model parameters:

num_units = [256, 128]
dropout_rate = [0.1, 0.2]
optimizer = ['sgd', 'adam']

Saves model architecture and model history recording training loss, validation loss, training accuracy and validation accuracy.

OBS: Outputs are saved in ../out. This directory should be made before running the utility script.
"""
# libraries
# tensorflow
import tensorflow as tf
from tensorflow.keras import layers

# tensorboard
import tensorboard
from tensorboard.plugins.hparams import api as hp


# plotting tools
import matplotlib.pyplot as plt
import numpy as np

# system tools
import os
from contextlib import redirect_stdout



# define class for grid search
class GridSearch:
    '''
    Grid Search object.
    Takes training data, validation data, image size and epochs as inputs.
    Defines hyperparameters, performs grid search and saves logs of hyperparameter tuning.
    '''
    def __init__(self, img_size, train_data, val_data, epochs):
        
        # defining changeable hyperparameters 
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([256, 128]))
        HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(.1, .2))
        HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

        # defining metrics to be recorded
        METRIC_ACCURACY = 'accuracy'

        # create file writer with hyperparameter configuration
        with tf.summary.create_file_writer('../logs/hparam_tuning').as_default():
            hp.hparams_config(
                hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
                metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
              )
        
        # training data
        self.train_data = train_data
        # validation data
        self.val_data = val_data
        # number of epochs
        self.epochs = epochs
        # image size
        self.img_size = img_size
        #log directory
        self.log_dir = '../logs/hparam_tuning'
        
        # defining number of classes to use for output layer
        self.num_classes = len(self.train_data.class_names)
        
        # starting at session 0
        session_num = 0

        # for loop for grid search
        
        # for the values of num_units
        for num_units in HP_NUM_UNITS.domain.values:
            # for the values of dropout rate
            for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                # for the values of optimizers
                for optimizer in HP_OPTIMIZER.domain.values:
                    # choose as hyperparameters
                    hparams = {
                        HP_NUM_UNITS: num_units,
                        HP_DROPOUT: dropout_rate,
                        HP_OPTIMIZER: optimizer,
                    }
                    # define run_name as current session
                    run_name = "run-%d" % session_num
                    # define run directory (changing directory at each run to avoid overwriting)
                    self.run_dir = '../logs/hparam_tuning/' + run_name
                    # print start of trial
                    print('--- Starting trial: %s' % run_name)
                    # print model parameters
                    print({h.name: hparams[h] for h in hparams})
                    
                    # save parameters for use in train_test_model and plot_history
                    num_units = hparams[HP_NUM_UNITS]
                    dropout = hparams[HP_DROPOUT] 
                    optimizer = hparams[HP_OPTIMIZER]
                    
                    # create summary log for model training
                    with tf.summary.create_file_writer(self.run_dir).as_default():
                        # record the values used in this trial
                        hp.hparams(hparams)  
                        # define, train and evaluate model with current hyperparameters
                        history, model_accuracy = train_test_model(self, num_units, dropout, optimizer, hparams)
                        # plot model history 
                        plot_history(self, history, num_units, dropout, optimizer)
                        
                        # create summary log for accuracy
                        tf.summary.scalar(METRIC_ACCURACY, model_accuracy, step=1)
                    
                    # update session number
                    session_num += 1
                


def train_test_model(self, num_units, dropout, optimizer, hparams):
    '''
    Function for training and testing model.
    Define model architecture with specified hyperparameters and save as .txt and .png.
    Train model and save model history.
    Evaluate model on validation data.
    '''
    # Model arcitecture
    # Sequential model
    model = tf.keras.models.Sequential([
        # input layer with rescaling 
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.img_size, self.img_size, 3)),
        # data augmentation: add random horizontal flip
        layers.experimental.preprocessing.RandomFlip("horizontal"), 
        # data augmentation: add random rotation
        layers.experimental.preprocessing.RandomRotation(0.1), 
        # first convolutional layer with 16 filters and relu activation
        layers.Conv2D(16, 3, padding='same', activation=tf.nn.relu),
        # max pooling
        layers.MaxPooling2D(),
        # second convolutional layer with 32 filters and relu activation
        layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu),
        # max pooling
        layers.MaxPooling2D(),
        # third convolutional layer with 64 filters and relu activation
        layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu),
        # max pooling
        layers.MaxPooling2D(),
        # flattening layer
        layers.Flatten(),
        # fully connected layer with changeable hyperparameter
        layers.Dense(num_units, activation=tf.nn.relu),
        # dropout layer
        layers.Dropout(dropout),
        # output layer with 98 nodes and softmax activation
        layers.Dense(self.num_classes, activation = 'softmax')
        ])
    
    # compile model with changeable optimizer
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  )
    
    # save model summary
    model_summary = model.summary()
    
    # name for saving model summary
    model_name = f"Architecture_{num_units}_{dropout}_{optimizer}.txt" 
    # path for saving model summary
    model_path = os.path.join("..", "out", model_name)
    # Save txt file
    with open(model_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    
    # name for saving plot
    plot_name = f"Architecture_{num_units}_{dropout}_{optimizer}.png" 
    # path for saving plot
    plot_path = os.path.join("..", "out", plot_name)
    # visualization of model summary plot
    model_plot = tf.keras.utils.plot_model(model,
                            to_file = plot_path,
                            show_shapes=True,
                            show_layer_names=True)
    
    
    # print that script has saved model architecture
    print(f"\n[INFO] Model architecture is saved as txt in '{model_path}' and as png in '{plot_path}'.")
    
    # fit model to data and save model history
    H = model.fit(self.train_data,
                  validation_data = self.val_data,
                  epochs=self.epochs, 
                  # defining callbacks (couldn't get tensorboard to work in my browser but might just be my computer and network)
                  callbacks=[tf.keras.callbacks.TensorBoard(self.run_dir, profile_batch=0),  # log metrics
                             hp.KerasCallback(self.run_dir, hparams)],) 

    # evaluate model with validation data and save model accuracy
    _, accuracy = model.evaluate(self.val_data)
    
    # save model
    model_name = f"model_{num_units}_{dropout}_{optimizer}"
    model_path = os.path.join("..", "out", model_name)
    model.save(model_path)

    return H, accuracy  

def plot_history(self, H, num_units, dropout, optimizer):
    """
    Plotting the loss/accuracy of the model during training and saving this as a png file in the out folder.
    """
    # name for saving output
    figure_name = f"History_{num_units}_{dropout}_{optimizer}.png"
    # path for saving output
    figure_path = os.path.join("..", "out", figure_name)
    # Visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, self.epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, self.epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, self.epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, self.epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    
    # print that script has saved
    print(f"\n[INFO] Loss and accuracy across on training and validation is saved as '{figure_path}'.")

