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

def evolution(x, u):
    _, _, theta, v, delta = x[:, 0]

    return np.array([[v*np.cos(delta)*np.cos(theta)],
                     [v*np.cos(delta)*np.sin(theta)],
                     [v*np.sin(delta)/3],
                     [u[0, 0]],
                     [u[1, 0]]])


# Create axes
ax = init_figure(-30, 40, -10, 40)

# Initialize the state
x = np.array([0, 0, 0, 10, 0.25])
x = x[:, None]

dt = 0.05

Galpha = np.zeros((5, 5))
Galpha[2, 2] = Galpha[3, 3] = Galpha[4, 4] = dt*0.01

nb_steps = 100

u = np.zeros((2, 1))

# Initialisation (a priori knowledge on x)
x_pred = np.array([[x[0, 0]],
                   [x[1, 0]],
                   [x[3, 0]]])
G_pred = np.zeros((3, 3))

Galpha_new = dt*0.01*np.eye(3)

Ak = np.eye(3)
uk = np.zeros((3, 1))

Gbeta = 0.01
Ck = np.array([[0, 0, 1]])

for k in range(nb_steps):
    clear(ax)
    
    # Draw the robot
    draw_car(x)
    
    yk = x[3] + mvnrnd1(Gbeta)
    
    Ak[0, 2] = dt*np.cos(x[4])*np.cos(x[2])
    Ak[1, 2] = dt*np.cos(x[4])*np.sin(x[2])
    
    uk[2, 0] = dt*u[0]
    
    x_pred, G_pred = Kalman(x_pred, G_pred, yk, Ck, Gbeta, uk, Ak, Galpha_new)
    
    # Draw the new confidence ellipse (on the predicted state)
    draw_ellipse_cov(ax, x_pred[:2], G_pred[:2, :2], 0.9, [1, 0.4, 0.4])
    
    # Add some noise
    alpha = np.random.multivariate_normal(np.zeros(5), Galpha).reshape(5, 1)
    
    # Discretized system with Euler method
    x = x + dt*evolution(x, np.zeros((2, 1))) + alpha
    
plt.show()
