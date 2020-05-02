import numpy as np
from numpy import exp, pi, cos, sin, r_
import math
import timeit
from scipy.signal import convolve2d, convolve

def convol2d(image, kernel):
    flipKernel = kernel.copy()
    for i in range(len(kernel)):
        flipKernel[i] = kernel[i][::-1]

    flipKernel = flipKernel[::-1]

    paddedImage = np.zeros((image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] -1))
    paddedImage[int(kernel.shape[0] / 2) : int(image.shape[0] + kernel.shape[0] / 2), int(kernel.shape[1] / 2) : int(image.shape[1] + kernel.shape[1] / 2)] = image
    filteredImage = np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filteredImage[i, j] = (flipKernel * paddedImage[i:i+kernel.shape[0], j:j+kernel.shape[1]]).sum()

    return filteredImage

def my_imfilter(image, kernel):
    """
    Your function should meet the requirements laid out on the project webpage.
    Apply a filter (using kernel) to an image. Return the filtered image. To
    achieve acceptable runtimes, you MUST use np multiplication and summation
    when applying the kernel.
    Inputs
    - image: np nd-array of dim (m,n) or (m, n, c)
    - kernel: np nd-array of dim (k, l)
    Returns
    - filtered_image: np nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    ##################
    # Your code here #
    dimensionProduct = kernel.shape[0] * kernel.shape[1]

    # Check if any of the dimensions is even
    if (dimensionProduct % 2) == 0:
        raise Exception('All the dimensions must be odd!')

    if len(image.shape) == 2:
        filteredImage = convol2d(image, kernel)
    else:
        trimShape = np.array([image.shape[0], image.shape[1]])

        redImage = np.zeros(trimShape)
        blueImage = np.zeros(trimShape)
        greenImage = np.zeros(trimShape)

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                redImage[i, j] = image[i, j, 0]
                greenImage[i, j] = image[i, j, 1]
                blueImage[i, j] = image[i, j, 2]

        filteredRedImage = convol2d(redImage, kernel)
        filteredGreenImage = convol2d(greenImage, kernel)
        filteredBlueImage = convol2d(blueImage, kernel)

        filteredImage = np.zeros(image.shape)
        
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                filteredImage[i, j] = [filteredRedImage[i, j], filteredGreenImage[i, j], filteredBlueImage[i, j]]

    return filteredImage
    ################


A = np.array([[1,2,3,4,5],
            [5,6,7,8,5],
            [9,10,11,12,4],
            [13,14,15,16,5],
            [1,1,1,1,1]])

B = np.array([[1,2,1,3,3],[3,2,1, 5,2],[4,4,4,2, 2],[3,3,2,2,1],[1,2,1,2,1]])

I = np.array([[1,2,3],[4,5,6],[7,8,9]])
J = np.array([[1,2,1],[2,3,2],[4,3,4]])

U = np.array([[1,2,3,4,5]])
V = np.array([1,2,3,4,5,6])

def fft(x):
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]

def fft2(spatialImage):
    nrows, ncols = spatialImage.shape

    fourierImage = np.zeros(spatialImage.shape, dtype = complex)
    P = np.zeros(spatialImage.shape, dtype=complex)

    for u in range(nrows):
        P[u] = fft(spatialImage[u,:])

    for v in range(ncols):
        fourierImage[:,v] = fft(P[:,v])

    return fourierImage



def transformFourier(spatialImage, inverse = False):
    m = spatialImage.shape[0]
    n = spatialImage.shape[1]

    xArray = np.array([i for i in range(0, m)])
    xExpoKernel = -(np.outer(xArray, xArray) * 1j * 2 * pi) / m
    xKernel = exp(xExpoKernel)
    yArray = np.array([i for i in range(0, n)])
    yExpoKernel = -(np.outer(yArray, yArray) * 1j * 2 * pi) / n
    yKernel = exp(yExpoKernel)

    fourierImage = np.zeros((m ,n), dtype=complex)
    
    for i in range(0, m):
        for j in range(0, n):
            transposeX = np.array([xKernel[i]]).T
            fourierImage[i, j] = sum(sum(spatialImage * (transposeX * yKernel[j])))

    return fourierImage  

def convol(image, kernel):
    nrows, ncols = (image.shape)

    padrows, padcols = (int(2**np.ceil(np.log2(nrows))), int(2**np.ceil(np.log2(ncols))))

    padimage = np.pad(image, ((0, padrows - nrows), (0, padcols - ncols)), mode = 'constant')
    padkernel = np.pad(kernel, ((0, padrows - kernel.shape[0]), (0, padcols - kernel.shape[1])), mode = 'constant')

    A = fft2(padimage)
    B = fft2(padkernel)

    return np.fft.ifft2(A*B).real[kernel.shape[0] // 2 : kernel.shape[0] // 2 + nrows, kernel.shape[1] // 2 : kernel.shape[1] // 2 + ncols]



print(convol(A, I))
print("================")
# A = np.pad(A, ((0,3),(0,3)), mode='constant')
# I = np.pad(I, ((0,5),(0,5)), mode='constant')
print(my_imfilter(A, I))
