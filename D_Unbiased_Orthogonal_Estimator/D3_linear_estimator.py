import numpy as np

# Input
# Measurement vector
y = np.array([[8],
              [7],
              [0]])

# System matrix
C = np.array([[2, 3],
              [3, 2],
              [1, -1]])

# Covariance matrix of the random noise
# (the 1st equation is supposed to be 2 times more accurate
# than the others)
Gbeta = np.array([[1, 0, 0],
                  [0, 4, 0],
                  [0, 0, 4]])

# Mean of the state (arbitrary)
x_mean = np.array([[0],
                   [0]])

# Covariance of the state
# (taken very large)
Gx = 1e6 * np.eye(2)

# To compute
# Innovation
y_tilde = y - np.dot(C, x_mean)

# Innovation covariance
Gy = np.dot(np.dot(C, Gx), np.transpose(C)) + Gbeta

# Kalman gain
K = np.dot(np.dot(Gx, np.transpose(C)), np.linalg.inv(Gy))

# Output
# Estimation
x_hat = x_mean + np.dot(K, y_tilde)
print(f"Estimated state x : {x_hat}")

# Uncertainty on the estimation
Gepsilon = Gx - np.dot(np.dot(K, C), Gx)
print(f"Uncertainty on the estimation : {Gepsilon}")

# Filtered measurements
y_hat = np.dot(C, x_hat)
print(f"Filtered measurements : {y_hat}")

# Residuals
r = y - y_hat
print(f"Residuals : {r}")
