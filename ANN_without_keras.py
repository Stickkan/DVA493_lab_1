import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


df = pd.read_csv('Diabetic.txt') # Reads the teext file and store it in dataframe (df)



data_set = df.values # Converting the dataframe (df) into an array

X = data_set[:,0:19]    # This is all the rows and every column except the last one
Y = data_set[:,19]      # This is all the rows but only the last column, the targets


min_max_scaler = preprocessing.MinMaxScaler()   # Scales the input features of X so that all input of the dataset lie within the range of 0 and 1
X_scale = min_max_scaler.fit_transform(X)       # This is combination with a sigmoid function or ReLU algorithm will decide if the node should fire or not



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)   # Splits the dataset into 75% training and 25% validation and test set

X_train = X_train.T
Y_train = Y_train.reshape(1, Y_train.shape[0])

X_test = X_test.T
Y_test = Y_test.reshape(1, Y_test.shape[0])



#X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.4)          # Splits the 25% into 10% validation and 15% into test set

print('Train X Shape: ', X_train.shape)
print('Train Y Shape: ', Y_train.shape)

def define_structure(X, Y):
    input_unit = X.shape[0]
    hidden_unit = 10
    output_unit = Y.shape[0]
    return (input_unit, hidden_unit, output_unit)

def parameters_init(input_unit, hidden_unit, output_unit):
    np.random.seed(2)
    W1 = np.random.randn(hidden_unit, input_unit)*0.01
    b1 = np.zeros((hidden_unit, 1))
    W2 = np.random.randn(output_unit, hidden_unit)*0.01
    b2 = np.zeros((output_unit, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = _sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache

def cross_entropy_cost(A2, Y, parameters):
    m = Y.shape[1] # number of training examples

    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1-A2))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))

    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1] # number of training examples
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis= 1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "dB2": db2}

    return grads

def gradient_descent(parameters, grads, learning_rate = 0.01):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "B2": b2}

    return parameters


def neural_network_model(X, Y, hidden_unit, num_iterations = 1000):
    np.random.seed(3)
    input_unit = define_structure(X, Y)[0]
    output_unit = define_structure(X, Y)[2]

    parameters = parameters_init(input_unit, hidden_unit, output_unit)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = cross_entropy_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = gradient_descent(parameters, grads)

        if i % 5 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters



def prediction(parameters, X):
        A2, cache = forward_propagation(X, parameters)
        predictions = np.round(A2)

        return predictions

parameters = neural_network_model(X_train, Y_train, 10, num_iterations = 1000)

(input_unit, hidden_unit_1, output_unit) = define_structure(X_train, Y_train)

print("The size of the input layer is: ", str(input_unit))
print("The size of the first hidden layer is: ", str(hidden_unit_1))
print("The size of the output layer is: ", str(output_unit))


predictions_train = prediction(parameters, X_train)
print('Accuracy train: %d' % float((np.dot(Y_train, predictions_train.T) + np.dot(1- Y_train, 1-predictions_train.T))/float(Y_train.size)*100 + '%'))

predictions_test = prediction(parameters, X_test)
print('Accuracy train: %d' % float((np.dot(Y_test, predictions_test.T) + np.dot(1- Y_test, 1-predictions_test.T))/float(Y_test.size)*100 + '%'))




    

"""
input_vectors = np.array(
    [
        [3, 1.5, 4, 2],
        [2, 1, 3.2, 1.5],
        [4, 1.5, 4.5, 2.5],
        [3, 4, 2.1, 5.1],
        [3.5, 0.5, 0.7, 0.2],
        [2, 0.5, 8.1, 11.2],
        [5.5, 1, 7.6, 5.3],
        [1, 1, 1, 1],
        [5.8, 0.4, 4.6, 5.7],
    ]
)

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1])




learning_rate = 0.1                # If greater then the steps are to big and the output just oscillates.

neural_network = NeuralNetwork(learning_rate)

training_error = neural_network.train(X_training, Y_training, 10000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("Cumulative_error.png")                     #The picture depicts all the errors combined
"""