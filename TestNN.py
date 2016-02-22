import numpy as np
from scipy.io import loadmat

import SimpleNN

mat = loadmat('..\\ex4weights.mat')

s = SimpleNN.SimpleNN([400, 25, 10])

s.theta = [np.transpose(mat["Theta1"]), np.transpose(mat["Theta2"])]

data = loadmat("..\\ex4data1.mat")

x = data["X"]
y = data["y"]

(cost, grad) = s.computeCostGrad(x, y, 1)
