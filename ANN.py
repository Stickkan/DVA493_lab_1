# Libraries that is recommended
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn import preprocessing
import matplotlib as plt
import numpy as np
import os as path
from sklearn.model_selection import train_test_split
#Artificial neural network


# ! Processing the data for use in the ANN

#Open file on Viktors computer
#file = 'Diabetic.txt'
#file_path = path.join(path.dirname(path.realpath(__file__)), file)
#data = pd.read_csv(file_path, on_bad_lines='skip')

#Open file on Fredriks computer

#We should try to 
df = pd.read_csv('Diabetic.txt') # Reads the teext file and store it in dataframe (df)



data_set = df.values # Converting the dataframe (df) into an array

X = data_set[:,0:19]    # This is all the rows and every column except the last one
Y = data_set[:,19]      # This is all the rows but only the last column


min_max_scaler = preprocessing.MinMaxScaler()   # Scales the input features of X so that all input of the dataset lie within the range of 0 and 1
X_scale = min_max_scaler.fit_transform(X)       #This is combination with a sigmoid function or ReLU algorithm will decide if the node should fire or not



X_training, X_val_and_test, Y_training, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.25)   # Splits the dataset into 75% training and 25% validation and test set

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.4)          # Splits the 25% into 10% validation and 15% into test set

print(X_training.shape, X_val.shape, X_test.shape, Y_training.shape, Y_val.shape, Y_test.shape)         # Print the values of each set


# ! Setting up the architecture of the ANN

model = Sequential([
    Dense(10, activation='sigmoid', input_shape=(19,)),             #Defining the architecture according to the tip given by Mr. Xiong where there are 2 hidden layers with 10 nodes in each
    Dense(10, activation='sigmoid'),                                #Input layer has 18 nodes. The hidden layers are ReLU activated however the output is sigmoid activated
    Dense(1, activation='sigmoid')                                  #Dense() refers to a fully connected layer

])                                                                                      

#'sgd' stochastic gradient descent, 'binary_crossentropy' the loss function for outputs that take the values 1 or 0, 'accuracy' tracks the accuracy of the loss function

model.compile(optimizer='sgd', loss = 'binary_crossentropy', metrics=['accuracy'])                                       

history = model.fit(X_training, Y_training, batch_size= 862, epochs= 2000, validation_data=(X_val, Y_val))


# ! this is what needs to be done
# 1) Learn Git
# 2) How to load the dataset
# 3) Divide the dataset into subsets (training/validation/test)
# 4) Create nodes and layers (Ning: No more than 2 hidden layers with 10 nodes in each)
    #Every node should have the sigmoid function. If the value is 0 then the output will not "fire" the next nodes in the next layer
# 5) Forwardpropagation
# 6) Error/cost function
# 7) B(l)ackpropagation
# 8) When do we stop / when is enough enough.


# ? Rebuild the tensorflow with the appropriate compiler flags?!