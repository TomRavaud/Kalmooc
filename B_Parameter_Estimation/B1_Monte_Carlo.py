import numpy as np
import matplotlib.pyplot as plt

# Parameters' true values
a_true = 0.9
b_true = 0.75

# Measurements vector (output)
y = np.array([0, 1, 2.65, 4.885, 7.646, 10.882])
y = y[:, None]  # Add a new axis

# Error threshold on the outputs
# (must be a small positive number)
epsilon = 0.5

plt.figure("Parameters space")

plt.xlabel("a")
plt.ylabel("b")

# a and b are taken between 0 and 2
plt.xlim([0, 2])
plt.ylim([0, 2])

# Number of random draws of parameters
nb_points = 1000

# Dynamic system state equations :
# x(k+1) = A x(k) + B u(k) = A x(k) + B
# (for each k, u(k) = 1)
# y(k) = C x(k)

# Constant C matrix
C = np.array([1, 1])

for i in range(nb_points):
    # Uniform random draw of a and b (between 0 and 2)
    a, b = np.random.rand(2) * 2
    
    # Compute the corresponding A matrix
    A = np.array([[1, 0],
                  [a, 0.9]])
    
    # Compute the corresponding B matrix
    B = np.array([[b],
                  [1-b]])
    
    # Initialize x and ym
    x = np.zeros((2, 1))
    ym = np.zeros((6, 1))

    # Compute the 6 consecutive states
    for k in range(6):
        x, ym[k] = np.dot(A, x) + B, np.dot(C, x)
    
    # If a and b model the system well enough
    if np.max(np.abs(ym - y)) < epsilon:
        plt.plot(a, b, "ro")  # Plot a red filled circle
    else:
        plt.plot(a, b, "b*")  # Plot a blue star

plt.show()

# Question 3 : analytical (a, b) solutions
a = np.linspace(0, 2, 50)
b = 0.75/(0.1 + a)

plt.figure()
plt.plot(a, b)
plt.show()
