import numpy as np

M = np.array([[4, 0],
              [10, 1],
              [10, 5],
              [13, 5],
              [15, 3]])

# Measured motor angular rates
y = np.array([[5],
              [10],
              [8],
              [14],
              [17]])

# Compute the left pseudo inverse of M using the formula
K = np.dot(np.linalg.inv(np.dot(np.transpose(M), M)),
           np.transpose(M))

# Solve the linear least squares optimization problem
p_estimated = np.dot(K, y)
print(p_estimated)

# Estimate the angular rate for U = 20V and Tr = 10Nm
omega = np.dot(np.array([20, 10]), p_estimated)
print(omega)