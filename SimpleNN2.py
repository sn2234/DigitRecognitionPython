import numpy as np
import math as m
from scipy.special import expit
from scipy.optimize import minimize
import pickle

def sigmoidGradient(x):
    t = expit(x)
    return t*(1-t)

xlog = np.vectorize(lambda x: m.log(1e-17) if x == 0 else m.log(x))

class NeuralNetConfig:
    inputSize = 0
    hiddenSize = 0
    outputSize = 0

    def __init__(self, inputSize, hiddenSize, outputSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

def prepareLabels(y, numLabels):
    numberOfSamples = y.shape[0]
    yy = np.zeros((numberOfSamples, numLabels))

    for i in range(y.shape[0]):
        yy[i, int(y[i])] = 1
    
    return yy

def initRandomThetas(nn):
    epsilon_init = 0.12
    th1 = np.random.random((nn.hiddenSize, nn.inputSize + 1))*2*epsilon_init - epsilon_init
    th2 = np.random.random((nn.outputSize, nn.hiddenSize + 1))*2*epsilon_init - epsilon_init

    return (th1, th2)

def combineThetas(th1, th2):
    return np.concatenate((
            th1.flatten(),
            th2.flatten()
        ))

def splitThetas(nn, comb):
    th1_len = nn.hiddenSize * (nn.inputSize + 1)
    #th2_len = nn.outputSize * (nn.hiddenSize +1)

    th1 = comb[:th1_len].reshape((nn.hiddenSize, nn.inputSize + 1))
    th2 = comb[th1_len:].reshape((nn.outputSize, nn.hiddenSize + 1))

    return th1, th2

# L - number of samples
# ni - input layer size
# nh - hidden layer size
# no - output layer size (equals to number of labels)
# x => (L * ni)
# y => (L * 1)
# yy => (L * no)
# th1 => (nh * (ni+1))
# th2 => (no * (nh+1))

def forwardPropagation(nn, th1, th2, x):
    numberOfSamples = x.shape[0]

    ones = np.ones((numberOfSamples, 1)) # (L * 1)

    a1 = np.hstack((ones, x)) # (L * (ni+1))
    
    z2 = (th1 @ a1.T).T # (L * (nh))

    a2 = np.hstack((ones, expit(z2))) # (L * (nh+1))

    z3 = (th2 @ a2.T).T # (L * no)

    a3 = expit(z3) # (L * no)

    return a1, z2, a2, z3, a3

def computeCost(nn, th1, th2, x, y, reg_lambda):
    numberOfSamples = x.shape[0]

    yy = prepareLabels(y, nn.outputSize)

    a1, z2, a2, z3, a3 = forwardPropagation(nn, th1, th2, x)

    costPositive = (-yy) * xlog(a3)
    costNegative = (1 - yy) * xlog(1 - a3)

    reg = (reg_lambda/(2*numberOfSamples)) * (np.sum((th1[:, 1:])**2) + np.sum((th2[:, 1:])**2))

    cost = np.sum(costPositive - costNegative)/numberOfSamples + reg

    return cost

def computeGrad(nn, th1, th2, x, y, reg_lambda):
    numberOfSamples = x.shape[0]

    yy = prepareLabels(y, nn.outputSize)
    ones = np.ones((numberOfSamples, 1)) # (L * 1)

    a1, z2, a2, z3, a3 = forwardPropagation(nn, th1, th2, x)

    delta3 = a3 - yy # (L * no)
    delta2 = (th2.T @ delta3.T).T * np.hstack((ones, sigmoidGradient(z2))) # (L * (nh+1))
    delta2 = delta2[:, 1:] # (L * (nh))

    theta1_reg = th1.copy()
    theta1_reg[:, 0] = np.zeros((th1.shape[0]))

    theta2_reg = th2.copy()
    theta2_reg[:, 0] = np.zeros((th2.shape[0]))

    thetaGrad1 = (delta2.T @ a1)/numberOfSamples + (reg_lambda/numberOfSamples) * theta1_reg
    thetaGrad2 = (delta3.T @ a2)/numberOfSamples + (reg_lambda/numberOfSamples) * theta2_reg

    return thetaGrad1, thetaGrad2

def predictProbability(nn, th1, th2, x):
    numberOfSamples = x.shape[0]

    _, _, _, _, a3 = forwardPropagation(nn, th1, th2, x.reshape((1, numberOfSamples)))

    return a3

def predictClass(nn, th1, th2, x):
    probabilities = predictProbability(nn, th1, th2, x).flatten()

    pmax = 0.0
    imax = 0
    for i in range(len(probabilities)):
        if probabilities[i] > pmax:
            pmax = probabilities[i]
            imax = i

    return imax

def computeCostComb(nn, thComb, x, y, reg_lambda):
    th1, th2 = splitThetas(nn, thComb)
    return computeCost(nn, th1, th2, x, y, reg_lambda)

def computeGradComb(nn, thComb, x, y, reg_lambda):
    th1, th2 = splitThetas(nn, thComb)
    th1p, th2p = computeGrad(nn, th1, th2, x, y, reg_lambda)
    return combineThetas(th1p, th2p)

def saveNetwork(nn, th1, th2, filePath):
    with open(filePath, "wb") as f:
        pickle.dump((nn, th1, th2), f)

def loadNetwork(filePath):
    with open(filePath, "rb") as f:
        (nn, th1, th2) = pickle.load(f)
        return (nn, th1, th2)

