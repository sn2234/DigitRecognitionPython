import numpy as np
import math as m
from scipy.special import expit

def sigmoidGradient(x):
    t = expit(x)
    return t*(1-t)

xlog = np.vectorize(lambda x: m.log(1e-17) if x == 0 else m.log(x))

class SimpleNN:
    #s = [784, 100, 10]
    s = [400, 25, 10]

    theta = []

    def setRandomWeights(self):
        self.theta = list((np.random.random((self.s[0]+1, self.s[1])), 
                     np.random.random((self.s[1]+1, self.s[2]))))

    def computeCostGrad(self, x, y, lmb):
        numberOfSamples = x.shape[0]

        # Transform y to 0/1 vector
        yy = np.zeros((numberOfSamples, self.s[2]))

        for i in range(y.shape[0]):
            yy[i, int(y[i]-1)] = 1

        # Compute activations
        a1 = np.hstack((np.ones((numberOfSamples, 1)), x))
        z2 = a1 @ self.theta[0]

        a2 = np.hstack((np.ones((numberOfSamples, 1)), expit(z2)))
        z3 = a2 @ self.theta[1]

        a3 = expit(z3)

        # Compute non-regularized cost
        xx = -1*(np.transpose(yy) @ xlog(a3)) - (np.transpose(1 - yy) @ xlog(1 - a3))

        # Compute regularization parameter
        reg = sum([np.sum(t[1:,:]*t[1:,:]) for t in self.theta])*lmb/(2*numberOfSamples)

        # Total cost
        cost = np.sum(xx)/numberOfSamples + reg

        # Compute backproparation gradients
        delta3 = a3 - yy
        delta2 = (
            np.transpose(self.theta[1] @ np.transpose(delta3)) * np.hstack((np.ones((numberOfSamples, 1)), sigmoidGradient(z2))))[:,1:]

        theta_reg1 = self.theta[0].copy()
        theta_reg1[:, 0] = np.zeros((theta_reg1.shape[0],))

        theta_reg2 = self.theta[1].copy()
        theta_reg2[:, 0] = np.zeros((theta_reg2.shape[0],))

        theta_grad = [
            (np.transpose(delta2)@a1)/numberOfSamples + (lmb/numberOfSamples)*np.transpose(theta_reg1),
            (np.transpose(delta3)@a2)/numberOfSamples + (lmb/numberOfSamples)*np.transpose(theta_reg2)]

        return (cost, theta_grad)

#s = SimpleNN()
#s.setRandomWeights()

#x = np.ones((1,784))
#y = np.ones((1,1))

#s.computeCostGrad(x, y, 0)
