# KNN_PCA_MNIST
This is an implementation of K-NN for classification of MNIST data set. We use Principal component analysis (PCA) to reduce the dimension of the data. 

## How to run
python3 main.py K, D, N_Testing, N_Training, PATH

K: Number of neighbors

D: Desired new dimension 

N_Testing: Number of testing points from data set

N_Training: Number of training points from data set 

PATH: Location of the data set

This will select the N_testing+N_training from data set and use the first N_testing for testing purpose and the other N_training for training purpose. Output is a text file showing the estimated label and its ground truth in each line. 
