import numpy as np
import random

A = np.array([[1,1,1],[2,2,2],[3,2.9,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9]])
B = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12]])
# np.random.shuffle(A[2:6])

X = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
Y = np.array([[1, 0, -4], [0, 1, -3], [0, 0, 1]])
Z = np.array([[3,4,1], [3, 4, 1],[3,4,1]])

print((X@Y@Z.T).T)