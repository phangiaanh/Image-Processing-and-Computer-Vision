import numpy as np

A = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]])
C = np.array([[1,1,1,1,0,0,0,0,1,1,1], [0,0,0,0,1,1,1,1,1,1,1],]*4)
D = np.array([2,1,2,3,4,5])
E = np.array([D,]*2).transpose()

Points_a = np.array([[1,2],[3,4],[5,6],[7,8]])
length = len(Points_a)
X = np.concatenate((Points_a, np.ones((length, 1)), Points_a, np.ones((length, 1)), Points_a), axis = 1)

Points = np.concatenate((np.repeat(Points_a, repeats = 3, axis = 1), np.ones((length, 2))), axis = 1)
print(X)
print(Points)