import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from pandas import read_csv as load_file


#Artificial neural network

#Loading the file, since we use pandas no need to open() or close()
##file = 'Diabetic.txt'
##file_name = path.join(path.dirname(path.realpath(__file__)), file)
##data = load_file(file_name)

#This is not the correct search way
#open('c:\Users\fredr\OneDrive\Dokument\Civilingenjör robotik\Pågående kurser\DVA493\Lab_1\Diabetic.txt', 'r')

#Open file on Viktors computer
file = 'Diabetic.txt'
file_path = path.join(path.dirname(path.realpath(__file__)), file)
data = pd.read_csv(file_path, on_bad_lines='skip')

#Open file on Fredriks computer


def training_set():
    #The training set is supposed to be 75% of entire set
    final_input_index = int(1150 * .75)
    return final_input_index

def validation_index(start_index):
    size_of_set = int(1150 * 0.1)
    last_index = start_index + size_of_set
    return last_index

def test_set():
    first_index = int(1150 * + .15)
    return first_index

print(f'The training set is from index 0 to {training_set()}\n')
print(f'The validation set is from index {training_set() + 1} to index {validation_index(training_set() + 1)}')
print(f'Lastly the test set is from {1150-(test_set() + 1)} to index 1150')
print(data)



#What needs to be done:
#1) Learn Git
#2) How to load the dataset
#3) Divide the dataset into subsets (training/validation/test)
#4) Create nodes and layers (Ning: No more than 2 hidden layers with 10 nodes in each)
#5) Forwardpropagation
#6) Error/cost function
#7) B(l)ackpropagation
#8) When do we stop / when is enough enough.

#I will just add a comment to see if there is any difference...  there was a difference :-)