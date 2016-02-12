import csv
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

rawData = []

with open("..\\train.csv", "r") as f:
    reader = csv.reader(f)
    for i in range(10):
        rawData.append(next(reader))

tmp = np.array(rawData)

y = tmp[1:, 0].astype(float)
x = tmp[1:, 1:].astype(float)

plt.imshow((x/255)[0].reshape((28, 28)))
