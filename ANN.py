import numpy as nm
import matplotlib.pyplot as plt
from os import path
from pandas import read_csv as load_file


#Artificial neural network

#Loading the file, since we use pandas no need to open() or close()
##file = 'Diabetic.txt'
##file_name = path.join(path.dirname(path.realpath(__file__)), file)
##data = load_file(file_name)

#This is not the correct search way
data_set = open(r"Diabetic.txt", "r")

print(data_set.read())

# Returns the final index of the training set
def training_set():
    #The training set is supposed to be 75% of entire set
    final_index = int(1150 * .75)
    return final_index

# Takes the start index input and returns the last index of validation set
def validation_set(start_index):
    size_of_set = int(1150 * 0.1)
    last_index = start_index + size_of_set
    return last_index

# Return the index which corresponds to 15% of the entire dataset
def test_set():
    first_index = int(1150 * + .15)
    return first_index

print(f'The training set is from index 0 to {training_set()}')
print(f'The validation set is from index {training_set() + 1} to index {validation_set(training_set() + 1)}')
print(f'Lastly the test set is from {1150-(test_set() - 1)} to index 1150')


# ! this is what needs to be done
# 1) Learn Git
# 2) How to load the dataset
# 3) Divide the dataset into subsets (training/validation/test)
# 4) Create nodes and layers (Ning: No more than 2 hidden layers with 10 nodes in each)
# 5) Forwardpropagation
# 6) Error/cost function
# 7) B(l)ackpropagation
# 8) When do we stop / when is enough enough.