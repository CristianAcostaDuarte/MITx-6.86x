import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *


train_x, train_y, test_x, test_y = get_MNIST_data()
train_x = np.array([[ 1., 34., 96., 22., 84., 77., 47., 95., 49., 93., 36.],
 [ 1., 13., 67., 54., 73., 56., 66., 3., 10., 84., 86.],
 [ 1., 48., 88., 12., 16., 99., 93., 82., 94., 86.,44.],
 [1. ,51. ,32. ,8. ,34. ,46. ,7. ,56. ,17. ,29. ,78.],
 [1. ,75. ,45. ,10. ,31. ,97. ,24. ,9. ,76. ,63. ,33.],
 [1. ,27. ,29. ,68. ,64. ,83. ,61. ,25. ,31. ,50. ,36.] ,
 [1,36,4,51,36,70,51,94,44,4,27],
 [1,21,68,82,77,43,30,92,30,17,6],
 [1,50,19,73,71,49,87,95,81,17,50],
 [1,30,44,19,70,63,79,2,37,58,59]])

train_y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
theta = np.zeros([10, train_x.shape[1]])
lambda_factor=1.0e-4
temp_parameter=1
cost = compute_cost_function(train_x, train_y, theta,  lambda_factor=1.0e-4, temp_parameter=1)

n = train_x.shape[0]
k = theta.shape[0]
regularization = (lambda_factor/2)*np.sum(theta**2)
exp_term = np.exp((np.dot(theta, train_x.T)/temp_parameter))
loss =  -np.log(exp_term/(np.sum(exp_term, axis = 0)))
print(f"X is:{train_x} ")
print(f"Y is:{train_y} ")
print(f"Theta is:{theta} ")
print(f"n is:{n} ")
print(f"k is:{k} ")

print(f"regu : {regularization}")
print(f"exp : {exp_term}")
print(f"loss : {loss}")
print(cost)