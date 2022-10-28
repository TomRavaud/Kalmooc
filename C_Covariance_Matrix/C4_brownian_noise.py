import numpy as np
import matplotlib.pyplot as plt

from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py


def compute_vectors(dt, sigma_x, T):
    # Time discretization
    t = np.arange(0, T, dt)     
    kmax = len(t)
    
    # Signal x
    x = sigma_x * np.random.randn(1, kmax)
    
    # Brownian noise y (discrete integration of x)
    # (cumulative sum of x elements values)
    y = dt * np.cumsum(x)
    
    return t, x, y

# Question 1
# Duration of the simulation
T = 100
# Time step
dt = 0.01

# Standard deviation of x
sigma_x = 1.0

# Compute the signal and the noise
t, x, y = compute_vectors(dt, sigma_x, T)

# Remove an axis
x = x[0, :]

plt.figure()
# Plot the signal and the noise
plt.plot(t, x, "b.")
plt.plot(t, y, "r.")
plt.show()

# We can see the square root growth of the standard deviation
# by plotting several Brownian noise vectors y
plt.figure()
for i in range(100):
    t, _, y = compute_vectors(dt, sigma_x, T)
    plot(t, y, 'red')
plt.show()

# Question 2
# Define a constant (unscaled variance)
K = 1
sigma_x_scaled = np.sqrt(K/dt)

plt.figure()
for i in range(100):
    t, _, y = compute_vectors(dt, sigma_x_scaled, T)
    plot(t, y, 'red')
plt.show()
