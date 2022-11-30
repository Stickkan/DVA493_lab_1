import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))
    
    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        return layer_2

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)                                      #Derivative of x^2
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)
        
        derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
        derror_dweights = (derror_dprediction * dprediction_dlayer1 * dlayer1_dweights)
        
        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    # ! Understand what this segment is doing
    def train(self, input_vectors, targets, epochs):
        cumulative_errors = []
        for current_iteration in range(epochs):
            random_data_index = np.random.randint(len(input_vectors))                       #Takes a random index from within the training set

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            derror_dbias, derror_dweights = self._compute_gradients(input_vector, target)

            self._update_parameters(derror_dbias, derror_dweights)

            if current_iteration % 100 == 0:
                cumulative_error = 0

                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
            cumulative_errors.append(cumulative_error)
        
        return cumulative_errors


input_vectors = np.array(
    [
        [3, 1.5],
        [2, 1],
        [4, 1.5],
        [3, 4],
        [3.5, 0.5],
        [2, 0.5],
        [5.5, 1],
        [1,1]
    ]
)

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])

"""
df = pd.read_csv('Diabetic.txt') # Reads the teext file and store it in dataframe (df)



data_set = df.values # Converting the dataframe (df) into an array

X = data_set[:,0:19]    # This is all the rows and every column except the last one
Y = data_set[:,19]      # This is all the rows but only the last column


min_max_scaler = preprocessing.MinMaxScaler()   # Scales the input features of X so that all input of the dataset lie within the range of 0 and 1
X_scale = min_max_scaler.fit_transform(X)       # This is combination with a sigmoid function or ReLU algorithm will decide if the node should fire or not



X_training, X_val_and_test, Y_training, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.25)   # Splits the dataset into 75% training and 25% validation and test set

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.4)          # Splits the 25% into 10% validation and 15% into test set
"""

learning_rate = 0.01

neural_network = NeuralNetwork(learning_rate)

training_error = neural_network.train(input_vectors, targets, 10000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all trainging instances")
plt.savefig("cumulative_error.png")