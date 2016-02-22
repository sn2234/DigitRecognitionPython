
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn import svm

import DataModel
import SimpleNN

(x, y) = DataModel.loadData("..\\train.csv")

(x_train, x_cv, y_train, y_cv) = DataModel.splitData(x, y)

s = SimpleNN.SimpleNN([784, 100, 10])

s.setRandomWeights()



#clf = svm.SVC(kernel = "rbf", C=0.9)

#x_sub = x_train[:1000,:]
#y_sub = y_train[:1000]

#clf.fit(x_sub, y_sub)

#y_pred = clf.predict(x_cv)

#err_rate = np.mean([1 if a != b else 0 for (a,b) in zip(y_pred, y_cv)])
