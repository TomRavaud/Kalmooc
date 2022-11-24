import numpy as np
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

def ExtendedKalman(x_pred, G_pred, y, C, Gbeta, u, f, A, Galpha, dt):
    """Implement an iteration of the Extended Kalman filter
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
    x_pred = x_cor + f(x, u)*dt  # Prediction performed on the initial nonlinear model
    G_pred = np.dot(np.dot(A, G_cor), np.transpose(A)) + Galpha
    
    return x_pred, G_pred

def f(x, u):
    """State evolution

    Args:
        x (ndarray (4, 1)): state of the system
        u (double): command

    Returns:
        ndarray (4, 1): state derivative
    """
    s, theta, ds, dtheta = x[0,0], x[1,0], x[2,0], x[3,0]
    dds = (mr*np.sin(theta)*(g*np.cos(theta)- l*dtheta**2) + u) / (mc+mr*np.sin(theta)**2)
    ddθ= (np.sin(theta)*((mr + mc)*g - mr*l*dtheta**2*np.cos(theta)) + np.cos(theta)*u)/ (l*(mc+mr*np.sin(theta)**2))
    
    return np.array([[ds],[dtheta],[dds],[ddθ]])


# Set system's parameters
mc, l, g, mr = 5, 1, 9.81, 1
dt = 0.03

# Initialize the state
x = np.array([[0, 0.2, 0, 0]]).T


# Linearized system matrices
A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, mr*g/mc, 0, 0],
              [0, (mc+mr)*g/(l*mc), 0, 0]])

B = np.array([[0],
              [0],
              [1/mc],
              [1/(l*mc)]])

C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])


# Controller u = -Kx + hw
# Gain matrix obtained with pole placement
K = place_poles(A, B, [-2.0, -2.1, -2.2, -2.3]).gain_matrix

# Given setpoint matrix
E = np.array([[1, 0, 0, 0]])

# Compute the precompensator
h = -np.linalg.inv(np.dot(np.dot(E, np.linalg.inv(A-np.dot(B, K))), B))

# Cart desired position
# (can be constant or time dependent for trajectory planning)
w = 1


# Initialisation (prior knowledge on x)
x_pred = np.zeros((4, 1))
G_pred = np.eye(4)

# Covariance matrices
Galpha = (np.sqrt(dt)*10**(-3))**2*np.eye(4)
Gbeta = 10**(-3)*np.eye(2)


ax = init_figure(-3, 3, -3, 3)

for t in arange(0, 8, dt) :
    clear(ax)
    draw_invpend(ax,x)
    
    # Controller
    u = -np.dot(K, x_pred)[0, 0] + w*h[0, 0]
    
    # Measure equation
    # Add some noise
    beta = mvnrnd1(Gbeta)
    y = np.dot(C, x) + beta
    
    # (Extended) Kalman filter for observing the cart state
    x_pred, G_pred = Kalman(x_pred, G_pred, y, C, Gbeta, dt*np.dot(B, u), np.eye(4)+dt*A, Galpha)
    # x_pred, G_pred = ExtendedKalman(x_pred, G_pred, y, C, Gbeta, u, f, np.eye(4)+dt*A, Galpha, dt)

    # Evolution
    # Add some noise
    alpha = mvnrnd1(Galpha)
    x = x + dt*f(x,u) + alpha

pause(1)    
 