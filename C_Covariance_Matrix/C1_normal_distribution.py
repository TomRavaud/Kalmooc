import numpy as np
import matplotlib.pyplot as plt

# Colormap for surface plotting
from matplotlib import cm

# For 3d plots
from mpl_toolkits.mplot3d import Axes3D

def p(X0_grid, X1_grid, x_mean, Gx, n=1):
    """Compute probability distribution values of a 2D random normal vector
    at grid points"""
    # Center the grid around the mean of x
    X0_centered = X0_grid - x_mean[0]
    X1_centered = X1_grid - x_mean[1]
    
    # Compute the inverse matrix of Gx
    Gx_inv = np.linalg.inv(Gx)
    
    Q = Gx_inv[0, 0]*X0_centered**2\
        + 2*Gx_inv[0, 1]*np.multiply(X0_centered, X1_centered)\
        + Gx_inv[1, 1]*X1_centered**2
        
    return 1/(np.sqrt((2*np.pi)**n*np.linalg.det(Gx))) * np.exp(-1/2*Q)

# Sample a 2D area
X0 = np.linspace(-15, 15, 50)
X1 = np.linspace(-15, 15, 50)
X0_grid, X1_grid = np.meshgrid(X0, X1)

# Question 1
# Mean of the random normal vector x
x_mean = np.array([[1],
                   [2]])

# Covariance matrix of x
Gx = np.eye(2)

# Probability distribution values
px  = p(X0_grid, X1_grid, x_mean, Gx)

# Question 2
# Let A and b such that y = Ax + b
A = np.dot(np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)],
                     [np.sin(np.pi/6), np.cos(np.pi/6)]]),
           np.array([[1, 0],
                     [0, 2]]))
b = np.array([[-2],
              [5]])

# Compute y_mean and Gy from x_mean and Gx
y_mean = np.dot(A, x_mean) + b
Gy = np.dot(np.dot(A, Gx), np.transpose(A))

# Probability distribution values
py  = p(X0_grid, X1_grid, y_mean, Gy)

# Function 3d plotting
plt.figure("Surface")
ax2 = plt.axes(projection="3d")
surf = ax2.plot_surface(X0_grid, X1_grid, py, cmap=cm.coolwarm)
plt.show()

# 3d contours plotting
plt.figure("Iso-contours")
ax = plt.axes(projection="3d")
ax.contour3D(X0_grid, X1_grid, py)
plt.show()
