# Age Detection
**This self assigned project was developed as part of the spring 2021 elective course Cultural Data Science - Visual Analytics at Aarhus University.**

__Task:__ The task for this project is to investigate age detection and see if it is possible to train a model that can detect how old a person is from an image. 
Furthermore, the project include grid search to find the best hyperparameters for the model. 
In addition, the project trains the best found model and uses this for an unseen image. The unseen image can be one that contain background noise as a bounding box can be defined for cropping the image.
The cropped version is then used for age prediction. <br>

For this project, I used facial image data with age labels retrieved from Kaggle. The data can be found on this link: https://www.kaggle.com/frabbisw/facial-age and should be downloaded and placed in the data folder prior to running the script. The data is structured as folders for each age containing face images. <br>

The repository contains two scripts in the src folder. Both scripts can be run without inputs (as they have defaults) or the user can specify these. <br>
The first script (__grid_search.py__) is used to find the best hyperparameters for a convolutional neural network to predict age from facial images. 
It looks at number of nodes in the fully connected layer, dropout rate and optimizers and returns model architecture and history for each individual model. <br> 
The second script (__detect_my_age.py__) uses the best found hyperparameters and trains a convolutional neural network. 
Similarly, the script takes an image and a region of interest (containing a face in the image). 
It then crops and saves the image of the ROI and uses this for model prediction. 
The script returns the model architecture, history, classification report and label predictions for the unseen image. <br>

In the Github repo there is a folder in the data folder called test_data. In this, I have included two images:
1. One of myself called IMG_3864.JPG 
2. One of my family called DSC04818.JPG. <br>

These can be used to test the model (bounding box specifications for these images can be found in the top of the script).

__Dependencies:__ <br>
To ensure dependencies are in accordance with the ones used for the script, you can create the virtual environment ‘age_venv"’ from the command line by executing the bash script ‘create_age_venv.sh’. 
```
    $ bash ./create_age_venv.sh
```
This will install an interactive command-line terminal for Python and Jupyter as well as all packages specified in the ‘requirements.txt’ in a virtual environment. 
After creating the environment, it will have to be activated before running the scripts.

```    
    $ source age_venv/bin/activate
```
After running these two lines of code, the user can commence running the grid search and age detection script. 
As the scripts do not depend on each other the order of running is irrelevant. <br>

### How to run grid_search.py <br>
This script uses tensorflow keras and a path to a folder directory of image data to find the optimal parameters for a convolutional neural network that can predict age from face data. 
For more information on the model specifications and implementation of grid search see 'grid_search_util.py' in utils folder in the Github repository.

__Parameters:__ <br>
```
    path: str <path-to-folder-directory>, default = "../data/face_data/face_age"
    output: str <path-to-output>, default = "../out"
    epochs: int <number-of-epochs>, default = 20
    logdir: str <path-to-log-directory>, default = "../logs"


```
    
__Usage:__ <br>
```
    grid_search.py -p <path-to-folder-directory> -o <path-to-output-folder> -e <epochs> -l <log-directory>
```
    
__Example:__ <br>
```
    $ cd src
    $ python3 grid_search.py -p ../data/face_data/face_age -o ../out -e 20 -l ../logs

```


### How to run detect_my_age.py <br>

As with grid_search.py, this script uses a path to a folder directory where the labels are equal to the folder names. It trains a convolutional neural network on face data to classify these into ages. 
Similarly, the script takes an image and a region of interest (containing a face in the image). It then crops and saves the image of the ROI and uses this for model prediction. 

If using the same folder structure as in the repo, the script can be run from the terminal without specifying parameters.


__Parameters:__ <br>
```
    path2folder: str <path-to-folder-directory>, default = "../data/face_data/face_age"
    output: str <path-to-output>, default = "../out"
    epochs: int <number-of-epochs>, default = 20
    path2image: str <path-to-unseen-image>, default = "../data/test_data/IMG_3864.JPG"
    x_start: int <start-of-ROI-on-x-axis>, default = 1450
    x_end: int <end-of-ROI-on-x-axis>, default = 1900
    y_start: int <start-of-ROI-on-y-axis>, default = 600
    y_end: int <end-of-ROI-on-y-axis>, default = 1050 

```
    
__Usage:__ <br>
```
    detect_my_age.py -p <path-to-folder> -o <path-to-output> -e <epochs> -i <path-to-image>, -x_s <start-on-x-axis> -x_e <end-on-x-axis> -y_s <start-on-y-axis> -y_e <end-on-y-axis>
```
    
__Example:__ <br>
```
    $ cd src
    $ python3 detect_my_age.py -p ../data/face_data/face_age -o ../out -e 20 -i ../data/test_data/IMG_3864.jpg -x_s 1450 -x_e 1900 -y_s 600 -y_e 1050

```

The code has been developed in Jupyter Notebook and tested in the terminal on Jupyter Hub on worker02. I therefore recommend cloning the Github repository to worker02 and running the scripts from there. 

### Results:
The results of the grid search showed a clear overfitting of the model on the training data across hyperparameter tuning. 
Running for 100 epochs, all models had an accuracy over 80% on the training data (and some even exceeded an accuracy of 97%). 
However, on the validation data no model had accuracy higher than 15%. 
As the models overfit pretty quickly, I ran the grid search a second time with data augmentation and this time for only 20 epochs. 
Here, training accuracy stayed below 25% (lowest: 14%) and validation data below 19% (lowest: 12%). 
So, though the accuracy is much lower on the training data, the validation accuracy stays approximately the same (with a slight increase). 
One pattern that becomes clear is that the model performs best with Adam optimizer compared to stochastic gradient descent. 
Similarly, using 256 compared to 128 nodes slightly increased model performance. 
So, the best model in the grid search had 256 nodes in fully connected layer, a dropout rate of 0.2 and used Adam optimization. 
This model yielded an accuracy of 24% on the training data and 19% on the test data. <br>

For the detect_my_age script, I used a photo of me from when I was 21 (IMG_3864.JPG). 
After running the defined model for 20 epochs, the model predicts me to be 16 in the image (which is not too bad because I looked a bit young for my age). 
However, looking at the label prediction plot the model seems quite confused. 
It has two major spikes with approximately 20 years apart, so it might as well have labeled me 36 years old. 
Furthermore, the weighted average accuracy of the model was 16%.  <br>

The labels of the data were treated as discrete categories though the real nature of age is continuous. 
To improve the model accuracy, it might therefore be better to decrease the categories by making them into ranges instead (e.g., grouping together ages like [20-24] and [25-29]). 
Alternatively, one could expand the accuracy to contain two ages before and after the true age. 
Chances are that the model might not be completely off just because the accuracy score is bad. 
This is due to the model accuracy being calculated on the bases of correct predictions. 
If the model predictions are off by one year it will still be wrong. Therefore, this might also be interesting to take into account. <br>

For further development, I would also like to implement the google Blazeface model to detect faces instead of manually defining the region of interest. I believe this will make the project even more generalizable and easy to use.

