
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

import Train
import SimpleNN2

# Load image as grayscale
fileName = r"..\iphonepuzzle_a-600x893.jpg"

img = io.imread(fileName, True)

# Load neural network
(nn, th1, th2) = SimpleNN2.loadNetwork(r"..\networkParams.bin")

# Try 28*28 slices

sliceSizeX = 28
sliceSizeY = 28

xSlices = img.shape[1] // sliceSizeX
ySlices = img.shape[0] // sliceSizeY

for x in range(xSlices):
    for y in range(ySlices):
        xFr = x*sliceSizeX
        xTo = xFr + sliceSizeX

        yFr = y*sliceSizeY
        yTo = yFr + sliceSizeY

        subImage = img[yFr:yTo, xFr:xTo]

        #prob = SimpleNN2.predictProbability(nn, th1, th2, subImage.flatten()).flatten()
        cl = SimpleNN2.predictClass(nn, th1, th2, subImage.flatten())

        if cl == 3:
            plt.imshow(subImage.reshape((28,28)))
            break

#for x in range(xSlices):
#    for y in range(ySlices):
#        xFr = x*sliceSizeX
#        xTo = xFr + sliceSizeX

#        yFr = y*sliceSizeY
#        yTo = yFr + sliceSizeY

#        subImage = img[yFr:yTo, xFr:xTo]

#        ax = plt.subplot(ySlices, xSlices, y*xSlices+x+1)
#        plt.set_cmap('gray')
#        plt.axis('off')
#        ax.imshow(subImage.reshape((28,28)))

#plt.subplots_adjust(hspace=-0.85)
#plt.gcf().set_size_inches(9, 9)

#plt.show()
