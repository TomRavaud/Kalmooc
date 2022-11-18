import numpy as np
import matplotlib.pyplot as plt

from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py

def Kalman(x_pred, G_pred, y, C, Gbeta, u, A, Galpha):
    """Implement an iteration of the Kalman filter

    Args:
        x_pred (ndarray (n, 1)): previous state prediction
        G_pred (ndarray (n, n)): previous x covariance matrix prediction
        y (ndarray (m, 1)): current measurement
        C (ndarray (m, n)): current C matrix
        Gbeta (ndarray (m, m)): current beta covariance matrix
        u (ndarray (n, 1)): current input of the system
        A (ndarray (n, n)): current A matrix
        Galpha (ndarray (n, n)): current alpha covariance matrix

    Returns:
        ndarray (n, 1), ndarray (n, n): new state prediction,
        new x covariance matrix prediction
    """
    ## Intermediate calculations ##
    # Innovation
    y_tilde = y - np.dot(C, x_pred)
    # Innovation covariance
    S = np.dot(np.dot(C, G_pred), np.transpose(C)) + Gbeta
    # Kalman gain
    K = np.dot(np.dot(G_pred, np.transpose(C)), np.linalg.inv(S))
    
    ## Correction step ##
    x_cor = x_pred + np.dot(K, y_tilde)
    G_cor = G_pred - np.dot(np.dot(K, C), G_pred)
    
    ## Prediction step ##
    x_pred = np.dot(A, x_cor) + u
    G_pred = np.dot(np.dot(A, G_cor), np.transpose(A)) + Galpha
    
    return x_pred, G_pred

def y(d, a, u):
    return d + np.cross(u, a)

# Create axes
ax = plt.axes()
plt.xlim([-25, 30])
plt.ylim([-20, 25])
plt.xlabel("x1")
plt.ylabel("x2")

# We only have a measurement equation
A = np.eye(2)
u = np.zeros((2, 1))
Galpha = np.zeros((2, 2))

Gbeta = 1

# Initialisation (a priori knowledge on x)
x_pred = np.array([[1],
                   [2]])
G_pred = 100 * np.eye(2)

# Draw the initial confidence ellipse (90%)
draw_ellipse_cov(ax, x_pred, G_pred, 0.9, [0.4, 1, 1])

# Compute all the measurements
a_all = np.array([[2, 1],
                  [15, 5],
                  [3, 12]])
b_all = np.array([[15, 5],
                  [3, 12],
                  [2, 1]])
d_all = np.array([2, 5, 4])
u_all = (b_all-a_all) / np.linalg.norm(b_all - a_all, axis=1)[:, None]
y_all = y(d_all, a_all, u_all)

C_all = u_all[:, ::-1]
C_all[:, 0] *= -1

colors = np.array([[0.4, 0.4, 1],
                   [0.4, 1, 0.4],
                   [1, 0.4, 0.4]])

for i in range(3):
    y = y_all[i]
    C = C_all[i][None, :]
    
    x_pred, G_pred = Kalman(x_pred, G_pred, y, C, Gbeta, u, A, Galpha)
    
    # Draw the new confidence ellipse (on the predicted state)
    draw_ellipse_cov(ax, x_pred, G_pred, 0.9, colors[i])

print(f"Estimated state x : {x_pred}")
print(f"Uncertainty on the estimation : {G_pred}")

plt.show()
