import numpy as np
from scipy.io import loadmat
from scipy.optimize import check_grad

import SimpleNN

mat = loadmat('..\\ex4weights.mat')

s = SimpleNN.SimpleNN([400, 25, 10])

s.theta = [np.transpose(mat["Theta1"]), np.transpose(mat["Theta2"])]

data = loadmat("..\\ex4data1.mat")

x = data["X"][:1,:]
y = data["y"][:1,:]

#check_grad(func = lambda p: s.computeCost(s.combineTheta(p), x, y, 0.5),
#                grad = lambda p: s.computeGrad(s.combineTheta(p), x, y, 0.5),
#                x0 = s.combineTheta(s.theta))

#(cost, grad) = s.computeCostGrad(s.theta, x, y, 1)

#predictions = [s.predictClass(w) for w in x]

#err_rate = np.mean([1 if pred != check else 0 for (pred, check) in zip(y, predictions)])

#print("Error rate with pre-computed paramaters: {0}", err_rate)

s.train(x, y, 0.5)

