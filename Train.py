import numpy as np
from scipy.optimize import minimize

def trainSciPy(net, x, y, lmb):
    net.setRandomWeights()

    combinedTheta = net.combineTheta(net.theta)
    optimizedTheta = minimize(
        fun = lambda p: net.computeCost(p, x, y, lmb),
        x0 = combinedTheta,
        method = 'CG',
        jac = lambda p: net.computeGrad(p, x, y, lmb),
        #callback = lambda xk: print("Iteration complete!"),
        options={'disp': True}) #'maxiter' : 5, 'eps' : 1e-10, 'gtol' : 1e-10

    net.theta = net.splitTheta(optimizedTheta.x)

    return net

def trainGradientDescent(net, x, y, lmb):
    net.setRandomWeights()
    alpha = 10

    thetaTmp = net.theta

    (costBefore, grad) = net.computeCostGrad(thetaTmp, x, y, lmb)

    thetaTmp1 = net.theta - alpha*grad

    (costAfter, _) = net.computeCostGrad(thetaTmp1, x, y, lmb)

    if costAfter > costBefore: