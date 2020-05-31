import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist

def getGaussianFilter(size, sigma):
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]

    gaussianKernel = np.exp(-((x**2 + y**2) / (2.0 * (sigma ** 2)))) / (2.0 * np.pi * (sigma ** 2))
     
    return gaussianKernel / gaussianKernel.sum()

def getSobelFilter(axis = 1):
    if axis == 0:
        return np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    else:
        return np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

def getDistanceMatrix(features1, features2, featureNumber1, featureNumber2):
    expandedFeatures1 = np.repeat(features1, repeats = featureNumber2, axis = 0)
    expandedFeatures2 = np.tile(features2, (featureNumber1, 1))

    expandedFeatures1 = expandedFeatures1 - expandedFeatures2
    expandedFeatures1 = expandedFeatures1 ** 2
    return np.sqrt(expandedFeatures1.sum(axis = 1).reshape((featureNumber1, featureNumber2)))

def getNormalizedPatches(image, x, y, featureWidth = 16):
    rowNumber = image.shape[0]
    colNumber = image.shape[1]
    features = np.zeros((len(x), 256))
    offset = np.uint16(featureWidth / 2)
    
    for i in range(len(x)):
        xCenter = np.uint16(x[i])
        yCenter = np.uint16(y[i]) 
        if xCenter >= offset and xCenter <= rowNumber - 2 * offset and yCenter >= offset and yCenter <= colNumber - 2 * offset:
            leftBound = np.uint16(x[i]) - offset
            rightBound = np.uint16(x[i]) + offset
            topBound = np.uint16(y[i]) - offset
            bottomBound = np.uint16(y[i]) + offset
            patches = np.array(image[topBound : bottomBound, leftBound : rightBound])
            patches = np.reshape(patches, (1, featureWidth ** 2))
            patches = patches / np.linalg.norm(patches)
            features[i, :] = patches
    return features

def getScaleInvariantFeature(image, x, y, featureWidth = 16):
    rowNumber = image.shape[0]
    colNumber = image.shape[1]
    offset = np.uint16(featureWidth / 2)

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
    
    features = np.ones((len(x), 128))
    for i in range(len(x)):
        xCenter = np.uint16(x[i])
        yCenter = np.uint16(y[i]) 
        if xCenter >= offset and xCenter <= rowNumber - 2 * offset and yCenter >= offset and yCenter <= colNumber - 2 * offset:
            leftBound = np.uint16(x[i]) - offset
            rightBound = np.uint16(x[i]) + offset
            topBound = np.uint16(y[i]) - offset
            bottomBound = np.uint16(y[i]) + offset

            histogram = np.zeros((1, 128))
            for j in range(8):
                layerImage = np.array(imageOctave[j])
                featureWindow = np.array(layerImage[topBound : bottomBound, leftBound : rightBound])
                featureWindow = np.split(featureWindow, 4, axis=0)
                featureWindow = np.array([np.split(x, 4, axis = 1) for x in featureWindow])
                histogramPart = (np.array([[y.sum() for y in x] for x in featureWindow])).flatten()
                for k in range(16):
                    histogram[0, j + 8 * k] = histogramPart[k]
            histogram = histogram / np.linalg.norm(histogram)
            features[i, :] = histogram
    return features

def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions
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

    harrisMatrix = (derivativeXXImage * derivativeYYImage - derivativeXYImage ** 2) - lambdaValue * (derivativeXXImage + derivativeYYImage) ** 2
    
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
            if  A == B and A >= 0.1:
                localMaxima.extend([[localIndex // feature_width + i, localIndex % feature_width + j]])

    localMaxima = np.unique(localMaxima, axis = 0)
    x = [item[0] for item in localMaxima]
    y = [item[1] for item in localMaxima]
    xs = np.array(x)
    ys = np.array(y)

    return ys, xs


def get_features(image, x, y, feature_width):
    '''
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''
    # TODO: Your implementation here! See block comments and the project webpage for instructions
    # return getNormalizedPatches(image, x, y, feature_width)
    return getScaleInvariantFeature(image, x, y, feature_width)


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    featureNumber1 = im1_features.shape[0]
    featureNumber2 = im2_features.shape[0]
    
    distanceMatrix = getDistanceMatrix(im1_features, im2_features, featureNumber1, featureNumber2)
    # distanceMatrix = cdist(im1_features, im2_features, 'euclidean')

    indexSortedMatrix = np.zeros((featureNumber1, featureNumber2))

    indexSortedMatrix = np.argsort(distanceMatrix)
    
    nearestNeighborIndex = indexSortedMatrix[:, 0]
    secondNearestNeighborIndex = indexSortedMatrix[:, 1]

    nearestNeighbor = list(map(lambda x, y: x[y], distanceMatrix, nearestNeighborIndex))
    secondNearestNeighbor = list(map(lambda x, y: x[y], distanceMatrix, secondNearestNeighborIndex))

    confidences = np.array(list(map(lambda x, y: y / x if x != 0 else -1, nearestNeighbor, secondNearestNeighbor)))

    matches = np.array([[i, nearestNeighborIndex[i]] for i in range(featureNumber1)])

    return matches, confidences
