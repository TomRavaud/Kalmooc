import numpy as np
import matplotlib.pyplot as plt

# Number of points
n = 1000

# Mean of the Gaussian vector x
x_mean = np.array([[1],
                   [2]])

# Covariance matrix of the Gaussian vector x
Gx = np.array([[3, 1],
               [1, 3]])

# Generate x by applying an affine transform to a
# unit Gaussian random vector
x = x_mean + np.dot(np.sqrt(Gx), np.random.randn(2, n))

# Display the random point cloud
plt.figure("Gaussian point cloud")
plt.plot(x[0, :], x[1, :], "r.")
plt.plot(x_mean[0], x_mean[1], "ko")

# Question 2
x2 = np.linspace(-5, 8, 100)
x1 = 1/3 + (1/3)*x2
plt.plot(x1, x2, "b", label="x1_hat = f(x2)")

# Question 3
x1 = np.linspace(-5, 5, 100)
x2 = 5/3 + (1/3)*x1
plt.plot(x1, x2, "g", label="x2_hat = g(x1)")

plt.legend()
plt.show()