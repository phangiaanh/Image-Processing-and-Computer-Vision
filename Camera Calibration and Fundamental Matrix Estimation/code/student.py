# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech

import numpy as np
from random import sample
import copy


def normalize(in_points):
    points = copy.copy(in_points)
    length = len(points)
    meanCoor = np.mean(points, axis = 0)
    
    newX = points[:, 0] - meanCoor[0]
    newY = points[:, 1] - meanCoor[1]

    stdX = 1 / np.std(newX)
    stdY = 1 / np.std(newY)

    T1 = np.array([[stdX, 0, 0], [0, stdY, 0], [0, 0, 1]])
    T2 = np.array([[1, 0, -meanCoor[0]], [0, 1, -meanCoor[1]], [0, 0, 1]])

    A = np.concatenate((points, np.ones((length, 1))), axis = 1)

    K = T1 @ T2
    T = K @ (A.T)
    
    return T.T, K

# Returns the projection matrix for a given set of corresponding 2D and
# 3D points. 
# 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
# 'Points_3D' is nx3 matrix of 3D coordinate of points in the world
# 'M' is the 3x4 projection matrix
def calculate_projection_matrix(Points_2D, Points_3D):
    # To solve for the projection matrix. You need to set up a system of
    # equations using the corresponding 2D and 3D points:
    #
    #                                                     [M11       [ u1
    #                                                      M12         v1
    #                                                      M13         .
    #                                                      M14         .
    #[ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
    #  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
    #  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
    #  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
    #  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
    #                                                      M32         un
    #                                                      M33         vn ]
    #
    # Then you can solve this using least squares with the 'np.linalg.lstsq' operator.
    # Notice you obtain 2 equations for each corresponding 2D and 3D point
    # pair. To solve this, you need at least 6 point pairs. Note that we set
    # M34 = 1 in this scenario. If you instead choose to use SVD via np.linalg.svd, you should
    # not make this assumption and set up your matrices by following the 
    # set of equations on the project page. 
    #
    ##################
    # Your code here #
    length = len(Points_2D)
    
    newPoints_2D = Points_2D.reshape((2 * length, 1))
    newPoints_2D = np.repeat(newPoints_2D, repeats = 3, axis = 1)
    newPoints_3D = np.repeat(Points_3D, repeats = 2, axis = 0)
    
    coeffMatrix = np.concatenate((newPoints_3D, np.ones((2 * length, 1)), newPoints_3D, np.ones((2 * length, 1)), newPoints_3D), axis = 1)
    elimMatrix = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], ] * length)
    
    coeffMatrix = coeffMatrix * elimMatrix
    elimMatrix = np.concatenate((np.ones((2 * length, 8)), -newPoints_2D), axis = 1)
    
    coeffMatrix = coeffMatrix * elimMatrix

    projectionMatrix = np.linalg.lstsq(coeffMatrix, Points_2D.reshape((2 * length, 1)))
    projectionMatrix = np.concatenate((projectionMatrix[0].flatten(), [1]), axis = 0)
    projectionMatrix = np.array(projectionMatrix).reshape((3, 4))

    return projectionMatrix
    ##################
    

# Returns the camera center matrix for a given projection matrix
# 'M' is the 3x4 projection matrix
# 'Center' is the 1x3 matrix of camera center location in world coordinates
def compute_camera_center(M):
    ##################
    # Your code here #
    inverseMatrix = np.linalg.inv(M[0 : 3, 0 : 3])
    center = -inverseMatrix @ M[:, 3]
    
    return center
    ##################


# Returns the camera center matrix for a given projection matrix
# 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
# 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
# 'F_matrix' is 3x3 fundamental matrix
def estimate_fundamental_matrix(Points_a,Points_b):
    # Try to implement this function as efficiently as possible. It will be
    # called repeatly for part III of the project
    ##################
    # Your code here #
    length = len(Points_a)

    [Points_a, Ta] = normalize(Points_a)
    [Points_b, Tb] = normalize(Points_b)

    coeffMatrix = np.concatenate((Points_a, Points_a,Points_a[:,0 : 2]), axis = 1)
    elimMatrix = np.concatenate((np.repeat(Points_b[:, 0 : 2], repeats = 3, axis = 1), np.ones((length, 2))), axis = 1)
    coeffMatrix = coeffMatrix * elimMatrix

    fundamentalMatrix = np.linalg.lstsq(coeffMatrix, -np.ones((length, 1)))
    fundamentalMatrix = np.concatenate((fundamentalMatrix[0].flatten(), [1]), axis = 0)
    fundamentalMatrix = fundamentalMatrix.reshape((3, 3))
    
    return ((Tb.T) @ fundamentalMatrix @ (Ta))
    ##################

# Takes h, w to handle boundary conditions
def apply_positional_noise(points, h, w, interval=3, ratio=0.2):
    """ 
    The goal of this function to randomly perturbe the percentage of points given 
    by ratio. This can be done by using numpy functions. Essentially, the given 
    ratio of points should have some number from [-interval, interval] added to
    the point. Make sure to account for the points not going over the image 
    boundary by using np.clip and the (h,w) of the image. 
    
    Key functions include but are not limited to:
        - np.random.rand
        - np.clip

    Arugments:
        points :: numpy array ~
            - shape: [num_points, 2] ( note that it is <x,y> )
            - desc: points for the image in an array
        h :: int 
            - desc: height of the image - for clipping the points between 0, h
        w :: int 
            - desc: width of the image - for clipping the points between 0, h
        interval :: int 
            - desc: this should be the range from which you decide how much to
            tweak each point. i.e if interval = 3, you should sample from [-3,3]
        ratio :: float
            - desc: tells you how many of the points should be tweaked in this
            way. 0.2 means 20 percent of the points will have some number from 
            [-interval, interval] added to the point. 
    """
    ##################
    # Your code here #
    newPoints = copy.copy(points)
    length = len(points)
    noisyPlaces = int(length * ratio)

    random = (np.random.rand(noisyPlaces, 2) - 0.5) * 2 * interval
    offset = np.concatenate((random, np.zeros((length - noisyPlaces, 2))))
    np.random.shuffle(offset)
    
    newPoints = newPoints + offset
    return np.clip(newPoints, [0, 0], [w, h])
    ##################

# Apply noise to the matches. 
def apply_matching_noise(points, ratio=0.2):
    """ 
    The goal of this function to randomly shuffle the percentage of points given 
    by ratio. This can be done by using numpy functions. 
    
    Key functions include but are not limited to:
        - np.random.rand
        - np.random.shuffle  

    Arugments:
        points :: numpy array 
            - shape: [num_points, 2] 
            - desc: points for the image in an array
        ratio :: float
            - desc: tells you how many of the points should be tweaked in this
            way. 0.2 means 20 percent of the points will be randomly shuffled.
    """
    ##################
    # Your code here #
    newPoints = copy.copy(points)
    length = len(points)
    noisyPlaces = int(length * ratio)

    startNoisy = np.random.randint(0, length - noisyPlaces)
    np.random.shuffle(newPoints[startNoisy : startNoisy + noisyPlaces])

    return newPoints
    ##################


# Find the best fundamental matrix using RANSAC on potentially matching
# points
# 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
# matching points from pic_a and pic_b. Each row is a correspondence (e.g.
# row 42 of matches_a is a point that corresponds to row 42 of matches_b.
# 'Best_Fmatrix' is the 3x3 fundamental matrix
# 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
# of 'matches_a' and 'matches_b') that are inliers with respect to
# Best_Fmatrix.
def ransac_fundamental_matrix(matches_a, matches_b):
    # For this section, use RANSAC to find the best fundamental matrix by
    # randomly sampling interest points. You would reuse
    # estimate_fundamental_matrix() from part 2 of this assignment.
    # If you are trying to produce an uncluttered visualization of epipolar
    # lines, you may want to return no more than 30 points for either left or
    # right images.
    ##################
    # Your code here #
    ##################

    # Your ransac loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.

    length = len(matches_a)
    threshold = 0.1
    bestInliers = np.array([])
    bestInlierNumber = 0
    bestMatrix = np.zeros((3, 3))

    picATest = np.concatenate((matches_a, np.ones((length, 1))), axis = 1)
    picBTest = np.concatenate((matches_b, np.ones((length, 1))), axis = 1)

    for _ in range(1000):
        index = sample(range(length), 9)
        pic_a = matches_a[index]
        pic_b = matches_b[index]
        fundamentalMatrix = estimate_fundamental_matrix(pic_a, pic_b)
        
        testArray = np.sum(picBTest @ fundamentalMatrix * picATest, axis = 1)
        
        testIndex = abs(testArray) < threshold
        testSum = sum(testIndex)

        if testSum > bestInlierNumber:
            bestInlierNumber = testSum
            bestInliers = testIndex
            bestMatrix = fundamentalMatrix

    return bestMatrix, matches_a[bestInliers], matches_b[bestInliers]
