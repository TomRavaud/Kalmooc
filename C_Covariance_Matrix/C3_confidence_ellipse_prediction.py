import numpy as np
import matplotlib.pyplot as plt

from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py

# Question 1, 2, 3
# Set the number of points
nb_points = 100

# Generate a random normal vector centered on the origin
y = np.random.randn(2, nb_points)

# Define a random normal variable by its mean and covariance matrice
x_mean = np.array([[1],
                   [2]])

Gx = np.array([[3, 1],
               [1, 3]])

# Transform the Gaussian random vector y by an affine transformation
x = x_mean + np.dot(np.sqrt(Gx), y)

plt.figure()
ax = plt.axes()

# Plot the two Gaussian random vectors on the same figure
plt.plot(y[0, :], y[1, :], "r.")
plt.plot(x[0, :], x[1, :], "b*")

# Draw the ellipse which contains 90% of the points
# (True because x is Gaussian)
draw_ellipse_cov(ax, x_mean, Gx, 0.9, [0.4, 0.4, 1])
# Same with 99% of the poins
draw_ellipse_cov(ax, x_mean, Gx, 0.99, [0.6, 0.6, 1])
# Same with 99.9% of the points
draw_ellipse_cov(ax, x_mean, Gx, 0.999, [0.8, 0.8, 1])

# Question 4, 5
# Time step
dt = 0.01

nb_iterations = 10

# Matrices of the state equation
A = np.array([[0, 1],
              [-1, 0]])
B = np.array([[2],
              [3]])

# Discretization of the system
# x(t+dt) = x(t) + dt*(Ax(t) + Bu(t))
#         = (I + dt*A)x(t) + dt*Bu(t)
A_new = np.eye(2) + dt*A

for i in range(nb_iterations):
    # Make the particules cloud evolve
    # Broadcast dt*Bu(t) where u(t) = sin(t)
    x = np.dot(A_new, x) + dt*np.sin(i*dt)*B
    plot(x[0, :], x[1, :], "g.")
    
    # Evolution of confidence ellipses (faster than applying the system to
    # the whole points cloud, but only in the Gaussian case
    # with linear equations)
    # Apply the state equation to the x vector mean
    x_mean = np.dot(A_new, x_mean) + dt*sin(i*dt)*B
    # Update the covariance matrix
    Gx = np.dot(np.dot(A_new, Gx), A_new)
    draw_ellipse_cov(ax, x_mean, Gx, 0.9, [0.8, 0.8, 1])
    
plt.show()
