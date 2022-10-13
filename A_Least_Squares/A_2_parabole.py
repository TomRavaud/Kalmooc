import numpy as np
import matplotlib.pyplot as plt

# True parabole parameters
pv = np.array([[np.sqrt(2)],[-1],[1]])
print(pv)

# Time vector
t = np.array([[-3],[-1],[0],[2],[3],[6]])

# Quadratic function computed at each time step
yv = pv[0,0]*t**2 + pv[1,0]*t + pv[2,0]
# These values are rounded to make them inaccurate
# (simulate noise)
y = np.round(yv)
print(y)

plt.figure()

# "Measured" values in black
plt.plot(t, y, "k", label="Measured values")
# True values in red
plt.plot(t, yv, "r", label="True values")

M = np.array([[9, -3, 1],
              [1, -1, 1],
              [0, 0, 1],
              [4, 2, 1],
              [9, 3, 1],
              [36, 6, 1]])

# Compute the left pseudo inverse of M using the formula
# Note : it would be faster to compute the SVD of M and the inverse of
# the diagonal matrix
K = np.dot(np.linalg.inv(np.dot(np.transpose(M), M)),
           np.transpose(M))

# Solve the linear least squares optimization problem
p_estimated = np.dot(K, y)
print(p_estimated)

y_filtered = np.dot(M, p_estimated)
print(y_filtered)
plt.plot(t, y_filtered, "g", label="Filtered measurements")

plt.legend()
plt.show()
