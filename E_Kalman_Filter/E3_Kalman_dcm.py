import numpy as np
import matplotlib.pyplot as plt

from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py

from E1_three_equations import Kalman


ax = plt.axes()
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xlabel("x1")
plt.ylabel("x2")

A = np.eye(2)
u = np.zeros((2, 1))
Galpha = np.zeros((2, 2))

Gbeta = 9

# Initialisation (a priori knowledge on x)
x_pred = np.array([[1],
                   [-1]])
G_pred = 4 * np.eye(2)

# Draw the initial confidence ellipse (90%)
draw_ellipse_cov(ax, x_pred, G_pred, 0.9, [1, 1, 1])

# All the measurements
y_all = np.array([5, 10, 11, 14, 17])

C_all = np.array([[4, 0],
                  [10, 1],
                  [10, 5],
                  [13, 5],
                  [15, 3]])

colors = np.array([[0.4, 0.4, 1],
                   [0.4, 1, 0.4],
                   [1, 0.4, 0.4],
                   [0.5, 0.5, 0.5],
                   [0.8, 0.8, 0.8]])

for i in range(5):
    y = y_all[i]
    C = C_all[i][None, :]
    print(C)
    
    x_pred, G_pred = Kalman(x_pred, G_pred, y, C, Gbeta, u, A, Galpha)
    
    # Draw the new confidence ellipse (on the predicted state)
    draw_ellipse_cov(ax, x_pred, G_pred, 0.9, colors[i])

print(f"Estimated state x : {x_pred}")
print(f"Uncertainty on the estimation : {G_pred}")

plt.show()
