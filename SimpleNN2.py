import numpy as np
import math as m
from scipy.special import expit
from scipy.optimize import minimize

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
    th1 = np.random.random((nn.hiddenSize, nn.inputSize + 1)) - 0.5
    th2 = np.random.random((nn.outputSize, nn.hiddenSize + 1)) - 0.5

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
def computeCost(nn, th1, th2, x, y, reg_lambda):
    numberOfSamples = x.shape[0]

    yy = prepareLabels(y, nn.outputSize)

    ones = np.ones((numberOfSamples, 1)) # (L * 1)

    a1 = np.hstack((ones, x)) # (L * (ni+1))
    
    z2 = (th1 @ a1.T).T # (L * (nh))

    a2 = np.hstack((ones, expit(z2))) # (L * (nh+1))

    z3 = (th2 @ a2.T).T # (L * no)

    a3 = expit(z3) # (L * no)

    costPositive = (-yy) * xlog(a3)
    costNegative = (1 - yy) * xlog(1 - a3)

    reg = (reg_lambda/(2*numberOfSamples)) * (np.sum((th1[:, 1:])**2) + np.sum((th2[:, 1:])**2))

    cost = np.sum(costPositive - costNegative)/numberOfSamples + reg

    return cost
