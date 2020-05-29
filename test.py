import numpy as np
from cmath import sqrt
from scipy.spatial.distance import cdist
import scipy.ndimage.filters as ft 

def gaussianFilter(size, sigma):
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]

    gaussianKernel = np.exp(-((x**2 + y**2) / (2.0 * (sigma ** 2)))) / (2.0 * np.pi * (sigma ** 2))
     
    return gaussianKernel / gaussianKernel.sum()

def getDistanceMatrix(features1, features2, featureNumber1, featureNumber2):
    expandedFeatures1 = np.repeat(features1, repeats = featureNumber2, axis = 0)
    expandedFeatures2 = np.tile(features2, (featureNumber1, 1))

    subMatrix = expandedFeatures1 - expandedFeatures2
    squareMatrix = subMatrix ** 2
    return np.sqrt(squareMatrix.sum(axis = 1).reshape((featureNumber1, featureNumber2)))

def match_features(im1_features, im2_features):
    featureNumber1 = im1_features.shape[0]
    featureNumber2 = im2_features.shape[0]
    
    distanceMatrix = getDistanceMatrix(im1_features, im2_features, featureNumber1, featureNumber2)

    indexSortedMatrix = np.zeros((featureNumber1, featureNumber2))

    indexSortedMatrix = np.argsort(distanceMatrix)

    nearestNeighborIndex = indexSortedMatrix[:, 0]
    secondNearestNeighborIndex = indexSortedMatrix[:, 1]

    nearestNeighbor = list(map(lambda x, y: x[y], distanceMatrix, nearestNeighborIndex))
    secondNearestNeighbor = list(map(lambda x, y: x[y], distanceMatrix, secondNearestNeighborIndex))

    confidences = list(map(lambda x, y: y / x if x != 0 else -1, nearestNeighbor, secondNearestNeighbor))

    matches = [[i, nearestNeighborIndex[i]] for i in range(featureNumber1)]

    return indexSortedMatrix, secondNearestNeighborIndex
    return matches, confidences

A = np.array([[1,2,3,4,5,6,7,8],
[3,2,3,2,3,2,3,2]])
B = np.array([[1,1,1,1,1,1,1,1],
[4,4,4,4,7,7,7,7],
[3,4,5,6,6,5,4,3]])

C = np.array([1,2,3,5,4,7,6])
X = np.random.random((3,3))
Y = np.random.random((5,3))
# print(np.allclose(getDistanceMatrix(X,Y,X.shape[0],Y.shape[0]), cdist(X,Y)))
print(getDistanceMatrix(X,Y,X.shape[0],Y.shape[0]))
print(match_features(X, Y))