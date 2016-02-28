
import sys
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score


import DataModel
import SimpleNN

(x, y) = DataModel.loadData("..\\train.csv")

(x_train, x_cv, y_train, y_cv) = DataModel.splitData(x, y)

s = SimpleNN.SimpleNN([784, 50, 10])

x_sub = x_train[:500,:]
y_sub = y_train[:500]

regs = np.linspace(0, 10, 20)
reg_acc_cv = []
reg_acc_train = []
max_acc = 0
best_reg = 0

for r in regs:
    s.train(x_sub, y_sub, r)

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

#clf = svm.SVC(kernel = "rbf", C=0.9)

#x_sub = x_train[:1000,:]
#y_sub = y_train[:1000]

#clf.fit(x_sub, y_sub)

#y_pred = clf.predict(x_cv)

#err_rate = np.mean([1 if a != b else 0 for (a,b) in zip(y_pred, y_cv)])
