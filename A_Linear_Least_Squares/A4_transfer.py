import numpy as np

error  = 1e-3

M = np.array([[1, 0, -1 + error, 1],
              [2, 1, 1, -1],
              [-3, 2, -1, 1],
              [-7, -3, 1, -1],
              [-11, -7, -1, 1],
              [-16, -11, 1, -1]])

y = np.array([[-2],
              [3],
              [7],
              [11],
              [16],
              [36]])

# Compute the left pseudo inverse of M using the formula
K = np.dot(np.linalg.inv(np.dot(np.transpose(M), M)),
           np.transpose(M))
print(K)

# Solve the linear least squares optimization problem
p_estimated = np.dot(K, y)
print(p_estimated)
