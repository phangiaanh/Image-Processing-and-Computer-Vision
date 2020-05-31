import numpy as np
from cmath import sqrt
from scipy.spatial.distance import cdist
from scipy.signal import convolve2d
from skimage.feature import peak_local_max

def getGaussianFilter(size, sigma):
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]

    gaussianKernel = np.exp(-((x**2 + y**2) / (2.0 * (sigma ** 2)))) / (2.0 * np.pi * (sigma ** 2))
     
    return gaussianKernel / gaussianKernel.sum()

def getSobelFilter():
    return np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

def get_interest_points(image, feature_width):
    smoothSmallFilter = getGaussianFilter(3, 1)
    smoothLargeFilter = getGaussianFilter(9, 2)

    smoothImage = convolve2d(image, smoothSmallFilter, mode = 'same')
    derivativeXImage = convolve2d(smoothImage, getSobelFilter(axis = 0), mode = 'same')
    derivativeYImage = convolve2d(smoothImage, getSobelFilter(axis = 1), mode = 'same')

    derivativeXXImage = derivativeXImage * derivativeXImage
    derivativeYYImage = derivativeYImage * derivativeYImage
    derivativeXYImage = derivativeXImage * derivativeYImage

    derivativeXXImage = convolve2d(derivativeXXImage, smoothLargeFilter, mode = 'same')
    derivativeYYImage = convolve2d(derivativeYYImage, smoothLargeFilter, mode = 'same')
    derivativeXYImage = convolve2d(derivativeXYImage, smoothLargeFilter, mode = 'same')

    # Szeliski 4.9
    lambdaValue = 0.06

    harrisMatrix = derivativeXXImage * derivativeYYImage - derivativeXYImage ** 2 - lambdaValue * (derivativeXXImage + derivativeYYImage) ** 2
    
    suppressMatrix = np.zeros(image.shape)
    suppressMatrix[feature_width : image.shape[0] - feature_width, feature_width : image.shape[1] - feature_width] = 1

    localMaxima = []
    suppressHarrisMatrix = harrisMatrix * suppressMatrix
    for i in range(feature_width, image.shape[0] - feature_width):
        for j in range(feature_width, image.shape[1] - feature_width):
            localMatrix = suppressHarrisMatrix[i : i + feature_width, j : j + feature_width]
            localIndex = np.argmax(localMatrix)
            A = suppressHarrisMatrix[localIndex // feature_width + i, localIndex % feature_width + j] 
            B = suppressHarrisMatrix[i, j]
            if  A >= 0.05:
                localMaxima.extend([[localIndex // feature_width + i, localIndex % feature_width + j]])

    localMaxima = np.unique(localMaxima, axis = 0)
    x = [item[0] for item in localMaxima]
    y = [item[1] for item in localMaxima]
    xs = np.array(x)
    ys = np.array(y)

    return ys, xs