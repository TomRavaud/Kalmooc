import numpy as np

from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py


def g(i):
    """Measurement function
    """
    # Landmarks
    T = np.array([[10540, 10920, 13740, 17480, 30380, 36880, 40240, 48170, 51720, 52320, 52790, 56880],
                  [1, 2, 1, 0, 1, 5, 4, 3, 3, 4, 5, 1],
                  [52.42,12.47,54.40,52.68,27.73,26.98,37.90,36.71,37.37,31.03,33.51,15.05]])
    y = np.array([[pz[i]]])
    C = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    Gbeta = 0.1*np.eye(1)
    
    # If a landmark is detected
    if i in T[0]:
        j = list(T[0]).index(i)
        k, rk = np.int32(T[1,j]), T[2,j]
        R = eulermat(phi[i], theta[i], psi[i])
        yi = R@np.array([[0, -sqrt(rk**2-a[i]**2), -a[i]]]).T
        y = np.vstack((yi[0:2,:], y))
        Ci = np.hstack((np.eye(2,3), np.zeros((2, 2*k)), -np.eye(2), np.zeros((2, 12-2*(k+1)))))
        C = np.vstack((Ci,C))
        Gbeta = 0.1*np.eye(3)
        
    return y, C, Gbeta


# Load measured data
D = loadcsv("./H_Bayes_Filter/slam_data.csv")

# Parse the table
t, phi, theta, psi, vr, pz, a = D[:,0], D[:,1], D[:,2], D[:,3], D[:,4:7].T, D[:,7], D[:,8]

# Time step
dt = 0.1

ax = init_figure(-200, 900, -300, 800)
N = len(t)


def kalman_predictor():
    # Initialize the state
    xhat = np.zeros((15,1))
    Gx = np.diag([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])*10**6

    Galpha = np.diag([0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Evolution matrix
    A = np.eye(15)

    for i in range(N):
        R = eulermat(phi[i], theta[i], psi[i])
        v = vr[:,i].reshape(3,1)
        u = np.vstack((dt*R@v, np.zeros((12,1))))
        xhat, Gx = kalman_predict(xhat, Gx, u, Galpha, A)

        if i%300 == 0:
            draw_ellipse_cov(ax, xhat[0:2], Gx[0:2,0:2], 0.99, [0.4,0.4,1])
            pause(0.01)

# kalman_predictor()
            

def kalman_filter():
    # A priori knowledge on x
    xhat = np.zeros((15, 1))
    Gx = np.diag([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])*10**4
    
    Galpha = np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])*10**(-2)
    
    # Evolution matrix
    A = np.eye(15)

    for i in range(N):
        R = eulermat(phi[i], theta[i], psi[i])
        ui = dt*R@vr[:, i]
        ui = np.vstack((ui.reshape((-1, 1)), np.zeros((12, 1))))
        y, Ci, Gbeta = g(i)
        
        xhat, Gx = kalman(xhat, Gx, ui, y, Galpha, Gbeta, A, Ci)
        
        if i%300 == 0:
            # Draw covariance matrix on robot position
            draw_ellipse_cov(ax, xhat[:2, 0], Gx[:2, :2], 0.99, [1, 0.4, 0.4])
            
            for j in range(3, len(xhat), 2):
                # Draw covariance matrix on landmark position
                draw_ellipse_cov(ax, xhat[j:j+2, 0], Gx[j:j+2, j:j+2], 0.99, [0.4, 1, 0.4])
                
            pause(0.001)

# kalman_filter()


def kalman_smoother():
    x_forward = {}
    G_forward = {}
    
    uk = {}
    
    x_update = {}
    G_update = {}
    
    x_forward[0] = zeros((15, 1))
    G_forward[0] = diag([0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])*10**4
    
    Galpha = diag([0.01,0.01,0.01,0,0,0,0,0,0,0,0,0,0,0,0])
    
    # Evolution matrix
    A = np.eye(15)

    for i in range(N):
        uk_tmp = dt * eulermat(phi[i], theta[i], psi[i]) @ vr[:, i]
        uk[i] = np.vstack((uk_tmp.reshape((-1, 1)), np.zeros((12, 1))))        
        y, Ck, Gbeta = g(i)
        
        x_update[i], G_update[i] = kalman_correc(x_forward[i], G_forward[i], y, Gbeta, Ck)
        x_forward[i+1], G_forward[i+1] = kalman_predict(x_update[i], G_update[i], uk[i], Galpha, A)
    
        if i%300 == 0:
            draw_ellipse_cov(ax, x_forward[i][:2, 0],G_forward[i][:2, :2], 0.99, [1, 0.4, 0.4])
            
            for j in range(3, 15, 2):
                draw_ellipse_cov(ax, x_forward[i][j:j+2, 0], G_forward[i][j:j+2, j:j+2], 0.99, [0.4, 1, 0.4])
            pause(0.001)

    x_backward = {N-1:x_update[N-1]}
    G_backward = {N-1:G_update[N-1]}
    
    for i in range(N-2, -1, -1):
        J = G_update[i] @ A.T @ np.linalg.inv(G_forward[i+1])
        x_backward[i] = x_update[i] + J @ (x_backward[i+1]-x_forward[i+1])
        G_backward[i] = G_update[i] + J @ (G_backward[i+1]-G_forward[i+1]) @ J.T

        if i%300 == 0:
            draw_ellipse_cov(ax, x_backward[i][:2, 0],G_backward[i][:2, :2], 0.99, [0.4, 0.4, 1])
            
            for j in range(3, 15, 2):
                draw_ellipse_cov(ax, x_backward[i][j:j+2, 0], G_backward[i][j:j+2, j:j+2], 0.99, [1, 1, 1])
            pause(0.001)
            
kalman_smoother()
