import numpy as np

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

A = np.array([[1,2,3],[2,3,4],[4,5,6]])
B = np.array([[1,2,3,4,5],
            [6,7,8,9,10],
            [11,12,13,14,15],
            [16,17,18,19,20],
            [21,22,23,24,25]])

C = np.array([[1,2,3,4,5],
            [6,7,8,99,10],
            [11,12,13,14,15],
            [16,17,18,19,20],
            [21,22,23,24,25]])

print(B - 1)