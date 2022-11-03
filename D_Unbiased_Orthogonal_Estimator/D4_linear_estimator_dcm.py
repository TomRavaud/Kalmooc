import numpy as np

# Input
# Measurement vector
y = np.array([5, 10, 8, 14, 17])
y = y[:, None]

# System matrix
C = np.array([[4, 0],
              [10, 1],
              [10, 5],
              [13, 5],
              [15, 3]])


# Covariance matrix of the random noise
Gbeta = 9 * np.eye(5)

# Mean of the state
x_mean = np.array([[1],
                   [-1]])

# Covariance of the state
Gx = 4 * np.eye(2)

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
