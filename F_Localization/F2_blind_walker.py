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


# Create axes
ax = plt.axes()
plt.xlim([-2, 12])
plt.ylim([0, 2])
plt.xlabel("x1")
plt.ylabel("x2")

nb_steps = 20

u = np.ones((1, nb_steps))
u[0, 10:] *= -1

Galpha = np.array([[0, 0],
                   [0, 0.0001]])

y = np.zeros((1, 1))
Gbeta = np.array([1])  # Set to 1 to avoid divisions by zero
C = np.zeros((1, 2))

# Initialisation (a priori knowledge on x)
x_pred = np.array([[0],
                   [1]])
G_pred = np.array([[0, 0],
                   [0, 0.0004]])

# Draw the initial confidence ellipse (99%)
draw_ellipse_cov(ax, x_pred, G_pred, 0.99, [0.4, 1, 1])

colors = np.array([[0.4, 0.4, 1],
                   [0.4, 1, 0.4],
                   [1, 0.4, 0.4]])

stdev_x1 = []
determinant_G = []

for k in range(nb_steps):
    A = np.array([[1, u[0, k]],
                  [0, 1]])
    x_pred, G_pred = Kalman(x_pred, G_pred, y, C, Gbeta, np.zeros((2, 1)), A, Galpha)
    
    # Draw the new confidence ellipse (on the predicted state)
    draw_ellipse_cov(ax, x_pred, G_pred, 0.99, colors[k%3])
    
    stdev_x1.append(np.sqrt(G_pred[0, 0]))
    determinant_G.append(np.linalg.det(G_pred))

print(f"Estimated state x : {x_pred}")
print(f"Uncertainty on the estimation : {G_pred}")

plt.figure()
plt.plot(range(nb_steps), stdev_x1, "b.")
plt.title("Evolution of the uncertainty on x1")
plt.xlabel("k")
plt.ylabel("stdev(x1)")

plt.figure()
plt.plot(range(nb_steps), determinant_G, "r.")
plt.title("Covariance matrix determinant")
plt.xlabel("k")
plt.ylabel("det(G)")

plt.show()
