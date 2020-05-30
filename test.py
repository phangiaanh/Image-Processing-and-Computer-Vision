import numpy as np
from cmath import sqrt
from scipy.spatial.distance import cdist
from scipy.signal import convolve2d

def getGaussianFilter(size, sigma):
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]

    gaussianKernel = np.exp(-((x**2 + y**2) / (2.0 * (sigma ** 2)))) / (2.0 * np.pi * (sigma ** 2))
     
    return gaussianKernel / gaussianKernel.sum()

def getSobelFilter():
    return np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

def getScaleInvariantFeature(image, x, y, featureWidth = 16):
    rowNumber = image.shape[0]
    colNumber = image.shape[1]
    # features = np.zeros((len(x), 128))
    offset = np.uint16(featureWidth / 2)
    # cellWidth = np.uint16(featureWidth / 4)

    sobelFilter = getSobelFilter()
    gaussianFilter = getGaussianFilter(int(offset), int(offset))
    sobelFilterOctave = np.zeros((8, sobelFilter.shape[0], sobelFilter.shape[1]))
    imageOctave = np.zeros((8, rowNumber, colNumber))

    for i in range(8):
        sobelFilter = np.array([[sobelFilter[0, 1], sobelFilter[0, 2], sobelFilter[1, 2]],
                [sobelFilter[0, 0], sobelFilter[1, 1], sobelFilter[2, 2]],
                [sobelFilter[1, 0], sobelFilter[2, 0], sobelFilter[2, 1]]])

        sobelFilterOctave[i, :] = sobelFilter

    for i in range(8):
        imageOctave[i, :] = convolve2d(image, sobelFilterOctave[i], mode = 'same')

    imageOctave = [convolve2d(x, gaussianFilter, mode = 'same') for x in imageOctave]
    # for i in range(len(x)):

    
    return imageOctave

A = np.array(range(256)).reshape((16, 16))
B = np.array([1,2,3,4,5,6,7,8])
C = [[1,2,3,4],[4,4,4,4]]
print(np.array(C))
print(B[1::2])