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
    
def observation(x, lm):
    nb_lm = np.shape(lm)[1]
    
    y = np.zeros((nb_lm + 1, 1))
    C = np.zeros((nb_lm + 1, 3))
    
    # We know the robot's speed
    y[0] = x[3]
    C[0] = np.array([0, 0, 1])
    
    # Go through the landmarks
    for i in range(nb_lm):
        m = lm[:, i]  # Landmark i coordinates
        distance = np.linalg.norm(m - x[:2].flatten())
        delta = np.arctan2(m[1] - x[1], m[0] - x[0]) - x[2]
        
        if distance < 15:
            plt.plot(np.array([m[0], x[0]], dtype=object), np.array([m[1], x[1]], dtype=object), "r")
            
            y[i] = m[0]*np.sin(x[2] + delta) - m[1]*np.cos(x[2] + delta)
            C[i, :] = np.array([np.sin(x[2] + delta), -np.cos(x[2] + delta), 0], dtype=object)
            
    # Add some noise
    beta = np.ones(nb_lm + 1)
    Gbeta = np.diag(beta)
    y = y + np.random.multivariate_normal(np.zeros(nb_lm+1), Gbeta).reshape((nb_lm+1, 1))
    
    return y, C, Gbeta
            
def observation2(xa, xb):
    distance_robots = xb[:2, 0] - xa[:2, 0]
    phi = arctan2(distance_robots[1], distance_robots[0]) - xa[2, 0]

    if norm(distance_robots) < 15:
        plt.plot(np.array([xa[0, 0], xb[0, 0]], dtype=object), np.array([xa[1, 0], xb[1, 0]], dtype=object), "b")
        C = np.array([-np.sin(xa[2, 0] + phi), np.cos(xa[2, 0] + phi), 0,
                      np.sin(xa[2, 0] + phi), -np.cos(xa[2, 0] + phi), 0])
        Gbeta = 1
        y = mvnrnd1(Gbeta)

        return y, C, Gbeta
    
    return False

def observation_fusion(xa, xb, lm):
    # Localization of the robots with the landmarks
    ya, Ca, Gbeta_a = observation(xa, lm)
    yb, Cb, Gbeta_b = observation(xb, lm)
    
    y = np.vstack((ya, yb))
    Gbeta = block_diag(Gbeta_a, Gbeta_b)
    C = block_diag(Ca, Cb)
    
    if observation2(xa, xb):
        # Communication between the two robots
        yab, Cab, Gbeta_ab = observation2(xa, xb)
    
        y = np.vstack((y, yab))
        Gbeta = block_diag(Gbeta, Gbeta_ab)
        C = np.vstack((C, Cab))
    
    return y, C, Gbeta


# Create axes
ax = init_figure(-30, 40, -10, 40)

# Time step
dt = 0.05

nb_steps = 100

# Landmarks
lm = np.array([[0, 15, 30, 15],
               [25, 30, 15, 20]])

def one_robot():
    # Initialize the state
    x = np.array([0, 0, 0, 10, 0.25])
    x = x[:, None]

    u = np.zeros((2, 1))

    # Initialisation (a priori knowledge on x)
    z_pred = np.zeros((3, 1))
    G_pred = 1e3*np.eye(3)

    Galpha = 0.01*dt*np.eye(3)

    Ak = np.eye(3)
    uk = np.zeros((3, 1))

    for k in range(nb_steps):
        clear(ax)

        # Draw the robot
        draw_car(x)

        # Draw the landmarks
        plt.scatter(lm[0], lm[1])

        # Observation equation
        yk, Ck, Gbeta = observation(x, lm)

        Ak[0, 2] = dt*np.cos(x[4])*np.cos(x[2])
        Ak[1, 2] = dt*np.cos(x[4])*np.sin(x[2])

        uk[2, 0] = dt*u[0]

        z_pred, G_pred = Kalman(z_pred, G_pred, yk, Ck, Gbeta, uk, Ak, Galpha)

        # Draw the new confidence ellipse (on the predicted state)
        draw_ellipse_cov(ax, z_pred[:2], G_pred[:2, :2], 0.9, [1, 0.4, 0.4])

        # Evolution equation
        # Add some noise
        alpha = np.zeros((5, 1))
        alpha[[0, 1, 3], 0] = np.random.multivariate_normal(np.zeros(3), Galpha)

        x = x + dt*evolution(x, uk) + alpha

    plt.show()

def two_robots():
    # Initialize the state
    xa = np.array([0, 0, 0, 15, 0.25])
    xb = np.array([10, 10, 5, 10, 0.25])
    xa = xa[:, None]
    xb = xb[:, None]

    ua = ub = np.zeros((2, 1))

    # Initialisation (a priori knowledge on x)
    z_pred = np.zeros((6, 1))
    G_pred = 1e3*np.eye(6)

    # Galpha_a and Galpha_b point to the same memory location
    Galpha_a = Galpha_b = 0.01*dt*np.eye(3)
    Galpha = block_diag(Galpha_a, Galpha_b)
    
    # Distinct Ak_a and Ak_b
    Ak_a = np.eye(3)
    Ak_b = np.eye(3)
    
    uk = np.zeros((6, 1))
    
    # Initialize noise vector
    alpha_a = np.zeros((5, 1))
    alpha_b = np.zeros((5, 1))

    for k in range(nb_steps):
        clear(ax)

        # Draw the robots
        draw_car(xa)
        draw_car(xb)

        # Draw the landmarks
        plt.scatter(lm[0], lm[1])

        # Observation equation
        yk, Ck, Gbeta = observation_fusion(xa, xb, lm)

        # Fill and gather the matrices Ak_a and Ak_b
        Ak_a[0, 2] = dt*np.cos(xa[4])*np.cos(xa[2])
        Ak_a[1, 2] = dt*np.cos(xa[4])*np.sin(xa[2])
        Ak_b[0, 2] = dt*np.cos(xb[4])*np.cos(xb[2])
        Ak_b[1, 2] = dt*np.cos(xb[4])*np.sin(xb[2])
        Ak = block_diag(Ak_a, Ak_b)
        
        uk[2, 0], uk[5, 0] = dt*ua[0], dt*ub[0]

        z_pred, G_pred = Kalman(z_pred, G_pred, yk, Ck, Gbeta, uk, Ak, Galpha)

        # Draw the new confidence ellipses (on the predicted state)
        draw_ellipse_cov(ax, z_pred[:2], G_pred[:2, :2], 0.9, [1, 0.4, 0.4])
        draw_ellipse_cov(ax, z_pred[3:5], G_pred[3:5, 3:5], 0.9, [0.4, 1, 0.4])

        # Evolution equation
        # Add some noise
        alpha_a[[0, 1, 3], 0] = np.random.multivariate_normal(np.zeros(3), Galpha_a)
        alpha_b[[0, 1, 3], 0] = np.random.multivariate_normal(np.zeros(3), Galpha_b)

        xa = xa + dt*evolution(xa, ua) + alpha_a
        xb = xb + dt*evolution(xb, ub) + alpha_b
        
    plt.show()

two_robots()
