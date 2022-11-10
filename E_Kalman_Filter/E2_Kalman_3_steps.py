import numpy as np
import matplotlib.pyplot as plt

from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py

from E1_three_equations import Kalman


ax = plt.axes()
plt.xlim([-100, 100])
plt.ylim([-100, 100])
plt.xlabel("x1")
plt.ylabel("x2")

# alpha and beta are white Gaussian signals with a unitary covariance matrix
Galpha = np.eye(2)
Gbeta = 1

# C is constant
C = np.array([[1, 1]])

# Initialisation (a priori knowledge on x)
x_pred = np.array([[0],
                   [0]])
G_pred = 1e2 * np.eye(2)

# Draw the initial confidence ellipse (90%)
draw_ellipse_cov(ax, x_pred, G_pred, 0.9, [1, 1, 1])

# k = 0
# Prediction step input
A = np.array([[0.5, 0],
              [0, 1]])
u = np.array([[8],
              [16]])

# Correction state input
y = 7

x_pred, G_pred = Kalman(x_pred, G_pred, y, C, Gbeta, u, A, Galpha)

# Draw the new confidence ellipse (on the predicted state)
draw_ellipse_cov(ax, x_pred, G_pred, 0.9, [0.4, 0.4, 1])

# k = 1
A = np.array([[1, -1],
              [1, 1]])
u = np.array([[-6],
              [-18]])
y = 30

x_pred, G_pred = Kalman(x_pred, G_pred, y, C, Gbeta, u, A, Galpha)

# Draw the new confidence ellipse
draw_ellipse_cov(ax, x_pred, G_pred, 0.9, [1, 0.4, 0.4])

# k = 2
u = np.array([[32],
              [-8]])
y = -6

x_pred, G_pred = Kalman(x_pred, G_pred, y, C, Gbeta, u, A, Galpha)

print(f"Estimated state x : {x_pred}")
print(f"Uncertainty on the estimation : {G_pred}")

# Draw the new confidence ellipse
draw_ellipse_cov(ax, x_pred, G_pred, 0.9, [0.4, 1, 0.4])

plt.show()
