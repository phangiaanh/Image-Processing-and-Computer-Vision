# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale

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

def transformFFT(inArray):
    return np.fft.fft(inArray)
    length = len(inArray)

    if length <= 64: return transformDFT(inArray)
    
    evenArray = transformFFT(inArray[0::2])
    oddArray = transformFFT(inArray[1::2])
    expoCoeff = [exp( -2j * pi * k / length) * oddArray[k] for k in range(length // 2)]
    return [evenArray[k] + expoCoeff[k] for k in range(length//2)] + [evenArray[k] - expoCoeff[k] for k in range(length//2)]

def transformFFT2(inMatrix):
    nrows, ncols = inMatrix.shape

    fourierImage = np.zeros(inMatrix.shape, dtype = complex)
    firstDegreeImage = np.zeros(inMatrix.shape, dtype=complex)

    for u in range(nrows):
        firstDegreeImage[u] = transformFFT(inMatrix[u,:])
    
    for v in range(ncols):
        fourierImage[:,v] = transformFFT(firstDegreeImage[:,v])
    
    return fourierImage

def transformIFFT2(inMatrix):
    copyMatrix = inMatrix.copy()
    nrows, ncols = inMatrix.shape

    fourierImage = np.zeros(inMatrix.shape, dtype = complex)
    firstDegreeImage = np.zeros(inMatrix.shape, dtype=complex)

    for i in range(nrows):
        copyMatrix[i, 1:] = copyMatrix[i, 1:][::-1]

    for u in range(nrows):
        firstDegreeImage[u] = transformFFT(copyMatrix[u,:])

    for i in range(ncols):
        firstDegreeImage[1:, i] = firstDegreeImage[1:, i][::-1]

    for v in range(ncols):
        fourierImage[:,v] = transformFFT(firstDegreeImage[:,v])

    return fourierImage

def convol2dFFT(image, kernel):
    nrows, ncols = (image.shape)

    padrows, padcols = (int(2**np.ceil(np.log2(nrows) + 1)), int(2**np.ceil(np.log2(ncols) + 1)))

    padimage = np.pad(image, ((0, padrows - nrows), (0, padcols - ncols)), mode = 'constant')
    padkernel = np.pad(kernel, ((0, padrows - kernel.shape[0]), (0, padcols - kernel.shape[1])), mode = 'constant')

    fourierImage = transformFFT2(padimage)
    fourierKernel = transformFFT2(padkernel)
    
    result = transformIFFT2(fourierImage * fourierKernel).real[:, :] / (padrows * padcols)

    return result[kernel.shape[0] // 2 : kernel.shape[0] // 2 + nrows, kernel.shape[1] // 2 : kernel.shape[1] // 2 + ncols]

def transformDFT(spatialImage, inverse = False):
    m = len(spatialImage)

    xArray = np.array([i for i in range(0, m)])
    xExpoKernel = -(np.outer(xArray, xArray) * 1j * 2 * pi) / m
    xKernel = exp(xExpoKernel)

    # return xKernel

    fourierImage = np.array([(spatialImage * xKernel[i]).sum() for i in range(m)], dtype=complex)
    
    return fourierImage

def my_imfilter(image, kernel):
    """
    Your function should meet the requirements laid out on the project webpage.
    Apply a filter (using kernel) to an image. Return the filtered image. To
    achieve acceptable runtimes, you MUST use numpy multiplication and summation
    when applying the kernel.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
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
        reverseShape = np.array((image.shape[2], image.shape[0], image.shape[1]))

        filteredImageArray = np.zeros(reverseShape)

        for i in range(image.shape[2]):
            filteredImageArray[i] = convol2d(image[:, :, i], kernel)

        filteredImage = np.zeros(image.shape)
        
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                filteredImage[i, j] = [item[i, j] for item in filteredImageArray]

    return np.clip(filteredImage, a_min = 0, a_max = 1)
    ################

"""
EXTRA CREDIT placeholder function
"""

def my_imfilter_fft(image, kernel):
    """
    Your function should meet the requirements laid out in the extra credit section on
    the project webpage. Apply a filter (using kernel) to an image. Return the filtered image.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
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
        filteredImage = convol2dFFT(image, kernel)
    else:
        reverseShape = np.array((image.shape[2], image.shape[0], image.shape[1]))

        filteredImageArray = np.zeros(reverseShape)

        for i in range(image.shape[2]):
            filteredImageArray[i] = convol2dFFT(image[:, :, i], kernel)

        filteredImage = np.zeros(image.shape)
        
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                filteredImage[i, j] = [item[i, j] for item in filteredImageArray]

    return np.clip(filteredImage, a_min = 0, a_max = 1)
    ################


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
    kernel2 = np.array(probs)
    kernel2.shape = (1, kernel2.shape[0])
    kernel1 = np.array([[item] for item in probs])

    # Your code here:
    low_frequencies_inter = my_imfilter_fft(image1, kernel1)
    low_frequencies = my_imfilter_fft(low_frequencies_inter, kernel2) # Replace with your implementation


    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    low_frequencies_inter_im2 = my_imfilter_fft(image2, kernel1)
    low_frequencies_im2 = my_imfilter_fft(low_frequencies_inter_im2, kernel2) # Replace with your implementation
    high_frequencies = image2 - low_frequencies_im2


    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = high_frequencies + low_frequencies # Replace with your implementation

    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to 
    # gen_hybrid_image().
    # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
    # and all values larger than 1.0 to 1.0.

    return np.clip(low_frequencies, a_min = 0, a_max = 1), np.clip(high_frequencies, a_min = 0, a_max = 1), np.clip(hybrid_image, a_min = 0, a_max = 1)