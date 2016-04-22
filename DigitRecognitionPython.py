 
import sys
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


import DataModel
import Train
import NN_1HL
import SimpleNN2

def findBestRegularization(s, x_sub, y_sub):
    regs = np.linspace(0, 10, 20)
    reg_acc_cv = []
    reg_acc_train = []
    max_acc = 0
    best_reg = 0

    for r in regs:
        th1, th2 = Train.trainSciPy2(s, x_sub, y_sub, r)

        acc_cv = accuracy_score(y_cv, [SimpleNN2.predictClass(s, th1, th2, w) for w in x_cv])
        acc_train = accuracy_score(y_sub, [SimpleNN2.predictClass(s, th1, th2, w) for w in x_sub])
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

def test3():
    (x, y) = DataModel.loadData("..\\train.csv")

    (x_train, x_cv, y_train, y_cv) = DataModel.splitData(x, y)

    x_sub = x_train[:20000,:]
    y_sub = y_train[:20000]

    s = SimpleNN2.NeuralNetConfig(784, 70, 10)

    regLambda = 6.84
    #s = Train.trainGradientDescent(s, x_sub, y_sub, 5)
    th1, th2 = Train.trainSciPy2(s, x_sub, y_sub, regLambda)
    #th1, th2 = Train.trainGradientDescent2(s, x_sub, y_sub, 5)

    acc_cv = accuracy_score(y_cv, [SimpleNN2.predictClass(s, th1, th2, w) for w in x_cv])
    print("Accuracy on CV set: {0}".format(acc_cv))

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

    cost_t = s_t.function(s_t.pack_thetas(thetas[0].copy(), thetas[1].copy()), 784, 70, 10, x_sub, y_sub, 10)
    grad_t = s_t.function_prime(s_t.pack_thetas(thetas[0], thetas[1]), 784, 70, 10, x_sub, y_sub, 10)
    print(cost_t, np.sum(grad_t));

    cost_my = s_my.computeCost(s_my.combineTheta(thetas.copy()), x_sub, y_sub, 10)
    grad_my = s_my.computeGrad(s_my.combineTheta(thetas), x_sub, y_sub, 10)

    print(cost_my, np.sum(grad_my))

def compareImplementations2():
    (x, y) = DataModel.loadData("..\\train.csv")

    y = y.astype(int)

    (x_train, x_cv, y_train, y_cv) = DataModel.splitData(x, y)

    x_sub = x_train[:500,:]
    y_sub = y_train[:500]

    s_my = SimpleNN2.NeuralNetConfig(784, 70, 10)
    s_t = NN_1HL.NN_1HL(reg_lambda = 10, opti_method = 'CG')

    np.random.seed(123)

    thetas = [s_t.rand_init(784,70), s_t.rand_init(70, 10)]
    
    # Check costs
    cost_t = s_t.function(s_t.pack_thetas(thetas[0].copy(), thetas[1].copy()), 784, 70, 10, x_sub, y_sub, 10)
    print("Cost test: ", cost_t)

    cost_my = SimpleNN2.computeCost(s_my, thetas[0], thetas[1], x_sub, y_sub, 10)
    print("Cost my: ", cost_my)

    # Check gradients
    grad_t = s_t.function_prime(s_t.pack_thetas(thetas[0].copy(), thetas[1].copy()), 784, 70, 10, x_sub, y_sub, 10)
    print("Grad sum test: ", np.sum(grad_t))

    grad_my1, grad_my2 = SimpleNN2.computeGrad(s_my, thetas[0], thetas[1], x_sub, y_sub, 10)
    print("Grad sum my: ", np.sum(grad_my1) + np.sum(grad_my2))

def trainFullAndSave():
    (x, y) = DataModel.loadData("..\\train.csv")

    (x_train, x_cv, y_train, y_cv) = DataModel.splitData(x, y)

    s = SimpleNN2.NeuralNetConfig(784, 70, 10)

    regLambda = 6.84
    
    print("Training neural network on full dataset")
    #s = Train.trainGradientDescent(s, x_sub, y_sub, 5)
    th1, th2 = Train.trainSciPy2(s, x_train, y_train, regLambda)
    #th1, th2 = Train.trainGradientDescent2(s, x_sub, y_sub, 5)

    print("Training complete, checking accuracy on CV data")

    acc_cv = accuracy_score(y_cv, [SimpleNN2.predictClass(s, th1, th2, w) for w in x_cv])
    print("Accuracy on CV set: {0}".format(acc_cv))

    SimpleNN2.saveNetwork(s, th1, th2, "..\\NeuralNetwork.bin")

def makeTestPerdictions():
    x, _ = DataModel.loadData("..\\test.csv")
    s, th1, th2 = SimpleNN2.loadNetwork("..\\NeuralNetwork.bin")
    
    y = [SimpleNN2.predictClass(s, th1, th2, w) for w in x]
    
    with open("results.csv", "w") as f:
        imageId = 1
        f.write("ImageId,Label\n")
        for i in y:
            f.write("{0},{1}\n".format(imageId, i))
            imageId = imageId + 1
    
#test3()


#(x, y) = DataModel.loadData("..\\train.csv")

#(x_train, x_cv, y_train, y_cv) = DataModel.splitData(x, y)

#subsetLen = 200
#x_sub = x_train[:subsetLen,:]
#y_sub = y_train[:subsetLen]

#s = SimpleNN2.NeuralNetConfig(784, 70, 10)
#bestReg = findBestRegularization(s, x_sub, y_sub)

makeTestPerdictions()

#(x, y) = DataModel.loadData("..\\train.csv")

#(x_train, x_cv, y_train, y_cv) = DataModel.splitData(x, y)

#s = SimpleNN2.NeuralNetConfig(784, 70, 10)

#regLambda = 6.84
##s = Train.trainGradientDescent(s, x_sub, y_sub, 5)
#th1, th2 = Train.trainSGD(s, x_train, y_train, regLambda)
##th1, th2 = Train.trainGradientDescent2(s, x_sub, y_sub, 5)
#costFinal = SimpleNN2.computeCost(s, th1, th2, x_train, y_train, regLambda)
#print("Final cost: {0}".format(costFinal))

#acc_cv = accuracy_score(y_cv, [SimpleNN2.predictClass(s, th1, th2, w) for w in x_cv])
#print("Accuracy on CV set: {0}".format(acc_cv))
