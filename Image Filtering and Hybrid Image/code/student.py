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
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    print('my_imfilter_fft function in student.py is not implemented')
    ##################

    return filtered_image


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
    low_frequencies_inter = my_imfilter(image1, kernel1)
    low_frequencies = my_imfilter(low_frequencies_inter, kernel2) # Replace with your implementation


    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    low_frequencies_inter_im2 = my_imfilter(image2, kernel1)
    low_frequencies_im2 = my_imfilter(low_frequencies_inter_im2, kernel2) # Replace with your implementation
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