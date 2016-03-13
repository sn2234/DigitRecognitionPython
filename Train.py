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
    alpha = 2

    thetaTmp = net.theta
    costs = []

    while True:
        (costBefore, grad) = net.computeCostGrad(thetaTmp, x, y, lmb)

        thetaTmp1 = [thetaTmp[0] - alpha*grad[0], thetaTmp[1] - alpha*grad[1]]

        (costAfter, _) = net.computeCostGrad(thetaTmp1, x, y, lmb)

        skipUpdate = False
        if costAfter > costBefore:
            alpha = alpha / 1.01
            skipUpdate = True
            print("Decrease alpha due to cyclic behaviour")

        #    thetaTmp1 = [thetaTmp[0] + alpha*grad[0], thetaTmp[1] + alpha*grad[1]]
        #    (costAfter, _) = net.computeCostGrad(thetaTmp1, x, y, lmb)

        #    if costAfter > costBefore:
        #        alpha = alpha / 1.5
        #        skipUpdate = True
        #        print("Decrease alpha due to cyclic behaviour")
    
        #print("costAfter: {0}".format(costAfter))

        if not skipUpdate:
            costs.append(costAfter)
            thetaTmp = thetaTmp1

        if len(costs) > 0 and len(costs) % 10 == 0:
            print('Epoch', len(costs), 'with cost', costs[-1], 'and alpha', alpha)

        if len(costs) > 2 and abs(costs[-2] - costs[-1]) < 0.00001:
            if alpha < 0.02:
                break
            else:
                print("Decrease alpha due to close costs")
                alpha = alpha / 1.5

    net.theta = thetaTmp
    return net

