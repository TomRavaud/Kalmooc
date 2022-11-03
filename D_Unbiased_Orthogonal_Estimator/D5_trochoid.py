import numpy as np
import matplotlib.pyplot as plt

# Question 1
# Input
# Measurement vector
y = np.array([[0.38],
              [3.25],
              [4.97],
              [-0.26]])

# System matrix
C = np.array([[1, np.cos(1)],
              [1, np.cos(2)],
              [1, np.cos(3)],
              [1, np.cos(7)]])

# Covariance matrix of the random noise
# (the 1st equation is supposed to be 2 times more accurate
# than the others)
Gbeta = 0.01 * np.eye(4)

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

# Question 2
p1, p2 = x_hat

T = np.linspace(0, 20, 100)

# Compute the path of the mass from estimated p1 and p2
X = p1*T - p2*np.sin(T)
Y = p1 - p2*np.cos(T)

plt.figure("Trochoid")
plt.plot(X, Y)
plt.xlabel("x")
plt.xlabel("y")
plt.show()
