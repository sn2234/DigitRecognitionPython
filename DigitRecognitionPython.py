 
import sys
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score


import DataModel
import SimpleNN
import Train
import NN_1HL

def findBestRegularization(s, x_sub, y_sub):
    regs = np.linspace(0, 10, 20)
    reg_acc_cv = []
    reg_acc_train = []
    max_acc = 0
    best_reg = 0

    for r in regs:
        s = Train.trainSciPy(s, x_sub, y_sub, r)

        acc_cv = accuracy_score(y_cv, [s.predictClass(w) for w in x_cv])
        acc_train = accuracy_score(y_sub, [s.predictClass(w) for w in x_sub])
        reg_acc_cv.append(acc_cv)
        reg_acc_train.append(acc_train)

        if max_acc < acc_cv:
            max_acc = acc_cv
            best_reg = r


        print("Validating regularization parameter [{0}]; Train accuracy: [{1}] CV accuracy: [{2}]"
              .format(r, acc_train, acc_cv))

    print("Best reg param: {0} with accuracy on CV dataset: {1}".format(best_reg, max_acc))

    plt.plot(regs, reg_acc_cv);plt.plot(regs, reg_acc_train)
    plt.show()

    return best_reg

def test1():
    (x, y) = DataModel.loadData("..\\train.csv")

    (x_train, x_cv, y_train, y_cv) = DataModel.splitData(x, y)

    x_sub = x_train[:500,:]
    y_sub = y_train[:500]

    s = SimpleNN.SimpleNN([784, 70, 10])

    #s = Train.trainGradientDescent(s, x_sub, y_sub, 5)
    s = Train.trainSciPy(s, x_sub, y_sub, 5)
    acc_cv = accuracy_score(y_cv, [s.predictClass(w) for w in x_cv])
    print("Accuracy on CV set: {0}", acc_cv)

def test2():
    (x, y) = DataModel.loadData("..\\train.csv")

    y = y.astype(int)

    (x_train, x_cv, y_train, y_cv) = DataModel.splitData(x, y)

    x_sub = x_train[:500,:]
    y_sub = y_train[:500]

    s = NN_1HL.NN_1HL(reg_lambda = 1, opti_method = 'CG')
    s.fit(x_sub, y_sub)

    acc_cv = accuracy_score(y_cv, [s.predict(w) for w in x_cv])
    print("Accuracy on CV set: {0}", acc_cv)

def compareImplementations():
    (x, y) = DataModel.loadData("..\\train.csv")

    y = y.astype(int)

    (x_train, x_cv, y_train, y_cv) = DataModel.splitData(x, y)

    x_sub = x_train[:500,:]
    y_sub = y_train[:500]

    s_my = SimpleNN.SimpleNN([784, 70, 10])
    s_t = NN_1HL.NN_1HL(reg_lambda = 1, opti_method = 'CG')

    np.random.seed(123)

    thetas = [s_t.rand_init(784,70), s_t.rand_init(70, 10)]

    cost_t = s_t.function(s_t.pack_thetas(thetas[0].copy(), thetas[1].copy()), 784, 70, 10, x_sub, y_sub, 1)
#    grad_t = s_t.function_prime(s_t.pack_thetas(thetas[0], thetas[1]), 784, 70, 10, x_sub, y_sub, 1)
    print(cost_t);

    cost_my = s_my.computeCost(s_my.combineTheta(thetas.copy()), x_sub, y_sub, 1)
#    grad_my = s_my.computeGrad(s_my.combineTheta(thetas), x_sub, y_sub, 1)

    print(cost_my)


compareImplementations()

#bestReg = findBestRegularization(s, x_sub, y_sub)

#clf = svm.SVC(kernel = "rbf", C=0.9)

#x_sub = x_train[:1000,:]
#y_sub = y_train[:1000]

#clf.fit(x_sub, y_sub)

#y_pred = clf.predict(x_cv)

#err_rate = np.mean([1 if a != b else 0 for (a,b) in zip(y_pred, y_cv)])
