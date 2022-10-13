import numpy as np
import matplotlib.pyplot as plt

# Colormap for surface plotting
from matplotlib import cm

# For 3d plots
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return x * y

def grad_f(x, y):
    return y, x

def g(x, y):
    return 2*x**2 + x*y + 4*y**2 + y - x + 3

def grad_g(x, y):
    return 4*x + y - 1, x + 8*y + 1

x = y = np.linspace(-5, 5, 10)

# Make a  grid
x_grid, y_grid = np.meshgrid(x,y)

# Compute gradients
# grad_x, grad_y = grad_f(x_grid, y_grid)
grad_x, grad_y = grad_g(x_grid, y_grid)

# Gradient plotting
plt.figure("Gradient")
plt.quiver(x_grid, y_grid, grad_x, grad_y)
plt.show()

# Function images
# z = np.array(f(np.ravel(x_grid),
#                np.ravel(y_grid))).reshape(x_grid.shape)
z = np.array(g(np.ravel(x_grid),
               np.ravel(y_grid))).reshape(x_grid.shape)

# 3d contours plotting
plt.figure("Iso-contours")
ax = plt.axes(projection="3d")
ax.contour3D(x, y, z)
plt.show()

# Function 3d plotting
plt.figure("Surface")
ax2 = plt.axes(projection="3d")
surf = ax2.plot_surface(x_grid, y_grid, z)
plt.show()
