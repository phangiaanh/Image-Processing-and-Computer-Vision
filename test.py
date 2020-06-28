import numpy as np
import random

A = np.array([[1,1,1],[2,2,2],[3,2.9,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9]])
B = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12]])
# np.random.shuffle(A[2:6])

C = np.array([1,2,3,4,5,6,7])
length = len(B)
noisyPlaces = int(length * 0.4)

D = np.array([1,2,3,4,-5,6,-7,8,-9])
E = D % 2 == 0
print(sum(E))