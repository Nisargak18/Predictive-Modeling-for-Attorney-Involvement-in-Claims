"""Fred is a very predictable man. For instance, when he uses his laptop, all he does is watch TV shows. He keeps on watching TV shows until his battery dies. Also, he is a very meticulous man, i.e. he pays great attention to minute details. He has been keeping logs of every time he charged his laptop, which includes how long he charged his laptop for and after that how long was he able to watch the TV. Now, Fred wants to use this log to predict how long will he be able to watch TV for when he starts so that he can plan his activities after watching his TV shows accordingly.

Challenge

You are given access to Fred’s laptop charging log by reading from the file “trainingdata.txt”. The training data file will consist of 100 lines, each with 2 comma-separated numbers.

The first number denotes the amount of time the laptop was charged.
The second number denotes the amount of time the battery lasted.
The training data file can be downloaded here (this will be the same training data used when your program is run). The input for each of the test cases will consist of exactly 1 number rounded to 2 decimal places. For each input, output 1 number: the amount of time you predict his battery will last.

Sample Input

1.50

Sample Output

3.00"""

SOLUTION:



import math
import os
import random
import re
import sys
import pandas as pd
import numpy as np
from scipy import optimize
#import matplotlib.pyplot as plt
from io import StringIO


if __name__ == '__main__':
    timeCharged = float(input().strip())

    # Read the data using pandas
    data = pd.read_csv('trainingdata.txt', header=None, names=["TimeCharged", "TimeLasted"])
    
    # training data: Range when not fully charge range
    index_values = data.query("TimeCharged <= 4.01").index.tolist()
    x = data.TimeCharged[index_values]
    y = data.TimeLasted[index_values]

    # Equation to be solved from book https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter16.02-Least-Squares-Regression-Derivation-Linear-Algebra.html
    # y = b1 * f1(x)
   
    # A Matrix
    # Only one linear basis function used. Important: Method allows any kind and number of basis function, as long as beta params are constants
    # Basis function is  evaluated at the input measured values. In this case f1(x) = x 
    A = np.array(x).reshape(-1,1)

    # Y Matrix: Output measured values
    Y = np.array(y).reshape(-1,1) 

    # Here is where the magic happends: Least square regression to find beta parameters
    beta_params = np.linalg.inv(A.T @ A) @ A.T @ Y
    
    
   # Make prediction
    if timeCharged > 4.01:  # Battery fully charged
        print(8)  
    else:
        predicted_time = timeCharged *  beta_params[0][0]
        print(predicted_time)  # Output the predicted battery life
