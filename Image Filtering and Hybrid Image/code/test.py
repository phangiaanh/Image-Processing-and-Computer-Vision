import numpy as np
from numpy import pi, exp, sqrt
import math
import matplotlib
import matplotlib.pyplot as plt
from helpers import vis_hybrid_image, load_image, save_image

from student import my_imfilter, gen_hybrid_image

def fft(x):
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]

def transformFourier(spatialImage, inverse = False):
    m = spatialImage.shape[0]
    n = spatialImage.shape[1]

    xArray = np.array([i for i in range(0, m)])
    xExpoKernel = -(np.outer(xArray, xArray) * 1j * 2 * pi) / m
    xKernel = exp(xExpoKernel)
    yArray = np.array([i for i in range(0, n)])
    yExpoKernel = -(np.outer(yArray, yArray) * 1j * 2 * pi) / n
    yKernel = exp(yExpoKernel)

    fourierImage = np.array((m ,n))

    for i in range(0, m):
        for j in range(0, n):
            transposeX = np.array([xKernel[i]]).T
            fourierImage[i, j] = np.dot(spatialImage, (transposeX * yKernel[j]))

    return fourierImage 


# dog = load_image('../data/dog.bmp')
# cat = load_image('../data/cat.bmp')

# low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(dog, cat, 3)

# A,B,C = gen_hybrid_image(dog,cat,7)

cutoff_frequency = 3
s, k = cutoff_frequency, cutoff_frequency*2
probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
kernel2 = probs
kernel1 = np.array([[item] for item in probs])
kernel2.shape = (1, kernel2.shape[0])
kernel = np.outer(probs, probs)
# A = my_imfilter(dog, kernel1)
# B = my_imfilter(A, kernel2)
# print(B)

X = np.array([[1,2,3,4,5],
            [6,7,8,9,10],
            [11,12,13,14,15],
            [16,17,18,19,20],
            [21,22,23,24,25]])

# C = my_imfilter(X, kernel)
# Dx = my_imfilter(X, kernel1)
# D = my_imfilter(Dx, kernel2)

# print(C)
# print("AAAAA")
# print(D)

h = math.ceil(np.log2(X))
print(h)