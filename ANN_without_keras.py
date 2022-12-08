#Imported packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os as os 


#import PlotDrawer as plot
#import matlab

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('Diabetic.txt', skiprows = 24, header = None) # Reads the text file and store it in dataframe (df)

df = pd.get_dummies(df, columns=[0, 1, 18])

scaler = MinMaxScaler()

df[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]] = scaler.fit_transform(df[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]])

df.head()

x = df.drop(19, axis=1).values
y = df[[19]].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)   # Splits the dataset into 75% training and 25% validation and test set
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.6)
          # Splits the 25% into 10% validation and 15% into test set
# ! Own functions that will be called

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def mse(prediction, labels):            #Labels is the value dictating if the patient has diabetes or not
    N = labels.size
    mse = ((prediction - labels)**2).sum()/(N*2)
    return mse

def accuracy(prediction, labels):
    corr_prediction = prediction.round()==labels
    accuracy = corr_prediction.mean()
    return accuracy

# ! Fixed parameters for the ANN

learning_rate = 0.1
epochs = 100000 #Number of iterations
N = y_train.size

# Number of nodes in each layer
n_input = 22
n_hidden_layer = 10
n_output = 1

# ! Initialize weights

np.random.seed(1)

W1 = np.random.normal(scale= 0.5, size=(n_input, n_hidden_layer)) # Creates n_input * n_hidden number of weights
W2 = np.random.normal(scale= 0.5, size=(n_hidden_layer, n_output))

supervise_train = {'Mean_squared_error': [], 'Accuracy': []}
supervise_val = {'Mean_squared_error': [], 'Accuracy': []}

# ! Train the ANN

for epochs in range(epochs):
    #Forward propagation
    hidden_layer = sigmoid(np.dot(x_train, W1))
    output_layer = sigmoid(np.dot(hidden_layer, W2))
    

    # Supervising the mean square error and the accuracy for the training set for each iteration
    mean_square_error = mse(output_layer, y_train)
    acc = accuracy(output_layer, y_train)
    supervise_train['Mean_squared_error'].append(mean_square_error)
    supervise_train['Accuracy'].append(acc)

    #Backward propagation - difference
    d_output_layer = (output_layer - y_train) * output_layer * (1-output_layer) #Derivation of sigmoid
    d_hidden_layer= np.dot(d_output_layer, W2.T) * hidden_layer * (1-hidden_layer)
    

    #Changes to the weight
    W2 -= learning_rate * np.dot(hidden_layer.T, d_output_layer) / N      # Dot product of the tranpose of second hidden layer and derivative of output layer
    W1 -= learning_rate * np.dot(x_train.T, d_hidden_layer) / N    # Dot product of the transpose of first hidden layer and the derivative of second hidden layer 


    #Supervise the accuracy of the validation set for each epoch
    
    hidden_layer_val = hidden_layer
    output_layer_val = output_layer
    hidden_layer_val = sigmoid(np.dot(x_val, W1))
    output_layer_val = sigmoid(np.dot(hidden_layer_val, W2))

    mse_val = mse(output_layer_val, y_val)
    acc_val = accuracy(output_layer_val, y_val)
    supervise_val['Mean_squared_error'].append(mse_val)
    supervise_val['Accuracy'].append(acc_val)

supervise_train_df = pd.DataFrame(supervise_train)
supervise_val_df = pd.DataFrame(supervise_val)

# ! Plot the training graph

#print(supervise_train_df) 
#print(supervise_val_df)

fig, axes = plt.subplots(1, 2, figsize=(15,5))

supervise_train_df.Mean_squared_error.plot(ax=axes[0], title = 'Mean squared error')
supervise_val_df.Accuracy.plot(ax=axes[1], title = 'Accuracy')
plt.show()