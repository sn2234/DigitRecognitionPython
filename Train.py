import sys
import numpy as np
from scipy.optimize import minimize

import SimpleNN2

def trainSciPy(net, x, y, lmb):
    net.setRandomWeights()

    combinedTheta = net.combineTheta(net.theta)
    optimizedTheta = minimize(
        fun = lambda p: net.computeCost(p, x, y, lmb),
        x0 = combinedTheta,
        method = 'BFGS',
        jac = lambda p: net.computeGrad(p, x, y, lmb),
        #callback = lambda xk: print("Iteration complete!"),
        options={'disp': True}) #'maxiter' : 5, 'eps' : 1e-10, 'gtol' : 1e-10

    net.theta = net.splitTheta(optimizedTheta.x)

    return net

def trainSciPy2(netConfig, x, y, lmb):

    th1, th2 = SimpleNN2.initRandomThetas(netConfig)

    combinedTheta = SimpleNN2.combineThetas(th1, th2)

    optimizedTheta = minimize(
        fun = lambda p: SimpleNN2.computeCostComb(netConfig, p, x, y, lmb) ,
        x0 = combinedTheta,
        method = 'L-BFGS-B',
        jac = lambda p: SimpleNN2.computeGradComb(netConfig, p, x, y, lmb),
        #callback = lambda xk: print("Iteration complete!"),
        options={'disp': False}) #'maxiter' : 5, 'eps' : 1e-10, 'gtol' : 1e-10

    return SimpleNN2.splitThetas(netConfig, optimizedTheta.x)

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

def trainGradientDescent2(netConfig, x, y, lmb):
    
    th1, th2 = SimpleNN2.initRandomThetas(netConfig)

    alpha = 2.0
    costs = []

    while True:

        costBefore = SimpleNN2.computeCost(netConfig, th1, th2, x, y, lmb)
        grad1, grad2 = SimpleNN2.computeGrad(netConfig, th1, th2, x, y, lmb)

        th1p = th1 - alpha*grad1
        th2p = th2 - alpha*grad2
        
        costAfter = SimpleNN2.computeCost(netConfig, th1p, th2p, x, y, lmb)

        skipUpdate = False
        if costAfter > costBefore:
            alpha = alpha / 1.01
            skipUpdate = True
            print("Decrease alpha due to cyclic behaviour")

        if not skipUpdate:
            costs.append(costAfter)
            th1 = th1p
            th2 = th2p

        if len(costs) > 0 and len(costs) % 10 == 0:
            print('Epoch', len(costs), 'with cost', costs[-1], 'and alpha', alpha)

        if len(costs) > 2 and abs(costs[-2] - costs[-1]) < 0.00001:
            if alpha < 0.02:
                break
            else:
                print("Decrease alpha due to close costs")
                alpha = alpha / 1.5
    
    return th1, th2

def findOptimalAlpha(netConfig, theta1, theta2, x, y, lmb, grad1, grad2, alphaFrom, alphaTo):
    alphas = np.linspace(alphaFrom, alphaTo, 15)
    bestAlpha = 0
    bestCost = sys.float_info.max

    for a in alphas:
        theta1p = theta1 - a*grad1
        theta2p = theta2 - a*grad2

        cost = SimpleNN2.computeCost(netConfig, theta1p, theta2p, x, y, lmb)
        if cost < bestCost:
            bestCost = cost
            bestAlpha = a
    
    return bestAlpha


def trainSGD(netConfig, x, y, lmb):
    th1, th2 = SimpleNN2.initRandomThetas(netConfig)

    alpha = 0.1
    costs = []

    numSamples = x.shape[0]
    miniBatchSize = 200

    for i in range((numSamples-2)//miniBatchSize):

        fr = i*miniBatchSize
        to = (i+1)*miniBatchSize

        xi = x[fr:to,:]
        yi = y[fr:to]

        costBefore = 0.0

        if len(costs) > 0:
            costBefore = costs[-1]
        else:
            costBefore = SimpleNN2.computeCost(netConfig, th1, th2, xi, yi, lmb)

        grad1, grad2 = SimpleNN2.computeGrad(netConfig, th1, th2, xi, yi, lmb)

        alpha = findOptimalAlpha(netConfig, th1, th2, xi, yi, lmb, grad1, grad2, alpha/2, alpha*2)

        th1p = th1 - alpha*grad1
        th2p = th2 - alpha*grad2

        costAfter = SimpleNN2.computeCost(netConfig, th1p, th2p, xi, yi, lmb)

        if costAfter <= costBefore:
            costs.append(costAfter)
            th1 = th1p
            th2 = th2p
        #else:
        #    # Find optimal alpha in a wide range
        #    alpha = findOptimalAlpha(netConfig, th1, th2, xi, yi, lmb, grad1, grad2, alpha/50, alpha)
        #    th1p = th1 - alpha*grad1
        #    th2p = th2 - alpha*grad2

        #    costAfter = SimpleNN2.computeCost(netConfig, th1p, th2p, xi, yi, lmb)

        #    if costAfter <= costBefore:
        #        costs.append(costAfter)
        #        th1 = th1p
        #        th2 = th2p

        if len(costs) > 0 and len(costs) % 10 == 0:
            print('Epoch', len(costs), 'with cost', costs[-1], 'and alpha', alpha)

    return th1, th2
