import numpy as np
import math as m
from scipy.special import expit
from scipy.optimize import minimize

def sigmoidGradient(x):
    t = expit(x)
    return t*(1-t)

xlog = np.vectorize(lambda x: m.log(1e-17) if x == 0 else m.log(x))

class SimpleNN:
    s = []
    #s = [784, 100, 10]
    #s = [400, 25, 10]

    theta = []

    def __init__(self, layers):
        self.s = layers

    def setRandomWeights(self):
        self.theta = [np.random.random((self.s[0]+1, self.s[1])), 
                      np.random.random((self.s[1]+1, self.s[2]))]

    def computeCostGrad(self, theta, x, y, lmb):
        numberOfSamples = x.shape[0]

        # Transform y to 0/1 vector
        yy = np.zeros((numberOfSamples, self.s[2]))

        for i in range(y.shape[0]):
            yy[i, int(y[i]-1)] = 1

        # Compute activations
        a1 = np.hstack((np.ones((numberOfSamples, 1)), x))
        z2 = a1 @ theta[0]

        a2 = np.hstack((np.ones((numberOfSamples, 1)), expit(z2)))
        z3 = a2 @ theta[1]

        a3 = expit(z3)

        # Compute non-regularized cost
        xx = -1*(yy * xlog(a3)) - ((1 - yy) * xlog(1 - a3))

        # Compute regularization parameter
        reg = sum([np.sum(t[1:,:]*t[1:,:]) for t in theta])*lmb/(2*numberOfSamples)

        # Total cost
        cost = np.sum(xx)/numberOfSamples + reg

        # Compute backproparation gradients
        delta3 = a3 - yy
        delta2 = (
            np.transpose(theta[1] @ np.transpose(delta3)) * np.hstack((np.ones((numberOfSamples, 1)), sigmoidGradient(z2))))[:,1:]

        theta_reg1 = theta[0].copy()
        theta_reg1[:, 0] = np.zeros((theta_reg1.shape[0],))

        theta_reg2 = theta[1].copy()
        theta_reg2[:, 0] = np.zeros((theta_reg2.shape[0],))

        theta_grad = [
            np.transpose((np.transpose(delta2)@a1)/numberOfSamples + (lmb/numberOfSamples)*np.transpose(theta_reg1)),
            np.transpose((np.transpose(delta3)@a2)/numberOfSamples + (lmb/numberOfSamples)*np.transpose(theta_reg2))]

        return (cost, theta_grad)

    def predictProbability(self, x):
        numberOfSamples = x.shape[0]

        h1 = expit(
            np.transpose(
                np.vstack((np.ones((1,1)), x.reshape((numberOfSamples,1))))) @ self.theta[0])
        h2 = expit(np.hstack((np.ones((1, 1)), h1)) @ self.theta[1])

        return h2.reshape((h2.size,))

    def predictClass(self, x):
        prob = self.predictProbability(x)
        
        pmax = 0.0
        imax = 0
        for i in range(len(prob)):
            if prob[i] > pmax:
                pmax = prob[i]
                imax = i

        return imax+1

    def combineTheta(self, theta):
        return np.concatenate((
            theta[0].flatten(),
            theta[1].flatten()
            ))

    def splitTheta(self, combinedTheta):
        t1_len = (self.s[0]+1) * self.s[1]
        t1 = combinedTheta[:t1_len].reshape((self.s[0]+1, self.s[1]))

        t2_len = (self.s[1]+1) * self.s[2]
        t2 = combinedTheta[t1_len:].reshape((self.s[1]+1, self.s[2]))

        return [t1, t2]

    def computeCost(self, combinedTheta, x, y, lmb):
        th = self.splitTheta(combinedTheta)
        (cost, _) = self.computeCostGrad(th, x, y, lmb)
        #print("New cost: {0}".format(cost))
        return cost

    def computeGrad(self, combinedTheta, x, y, lmb):
        th = self.splitTheta(combinedTheta)
        (_, grad) = self.computeCostGrad(th, x, y, lmb)
        return self.combineTheta(grad)

    #def train(self, x, y, lmb):
    #    self.setRandomWeights()

    #    combinedTheta = self.combineTheta(self.theta)
    #    optimizedTheta = minimize(
    #        fun = lambda p: self.computeCost(p, x, y, lmb),
    #        x0 = combinedTheta,
    #        method = 'TNC',
    #        jac = lambda p: self.computeGrad(p, x, y, lmb),
    #        #callback = lambda xk: print("Iteration complete!"),
    #        options={'disp': False}) #'maxiter' : 5, 'eps' : 1e-10, 'gtol' : 1e-10

    #    self.theta = self.splitTheta(optimizedTheta.x)

    #    return self.theta

#s = SimpleNN()
#s.setRandomWeights()

#x = np.ones((1,784))
#y = np.ones((1,1))

#s.computeCostGrad(x, y, 0)
