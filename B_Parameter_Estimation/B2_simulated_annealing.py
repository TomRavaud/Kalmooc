from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py

import numpy as np

def draw_room():
    """Draw the room
    """
    for j in range(A.shape[1]):
        plot(array([A[0,j],B[0,j]]),array([A[1,j],B[1,j]]),color='blue')
        
def draw(p,y,col):
    """Draw the robot and the laser

    Args:
        p (ndarray (3, 1)): pose of the robot
        y (ndarray (8, 1)): length of the rays
        col (string): color of the rays
    """
    draw_tank(p,'darkblue',0.1)
    p=p.flatten()
    y=y.flatten()
    for i in arange(0,8):
        plot(p[0]+array([0,y[i]*cos(p[2]+i*pi/4)]),p[1]+array([0,y[i]*sin(p[2]+i*pi/4)]),color=col)

def f(p, A, B):
    """Compute laser rays' length

    Args:
        p (ndarray (3, 1)): pose of the robot
        A (ndarray (2, n)): coordinates of the first extremity of walls segments
        B (ndarray (2, n)): coordinates of the second extremity of walls segments

    Returns:
        ndarray (8, 1): rays' length
    """
    # Initialize the vector which will contain the distances
    y = np.ones((8, 1))*1e8
    
    # For each ray
    for k in range(8):
        # Compute the direction of the ray
        u = np.array([[np.cos(p[2, 0] + np.pi/4*k)],
                      [np.sin(p[2, 0] + np.pi/4*k)]])
        # Get the current position of the robot on the map
        m = np.array([[p[0, 0]],
                      [p[1, 0]]])
        
        # Go through the list of walls
        for i in range(np.shape(A)[1]):
            # Get the extremities of the wall i
            a, b = A[:, i, None] , B[:, i, None]
            
            # Intersection condition
            if np.linalg.det(np.concatenate([a-m, u], axis=1))*np.linalg.det(np.concatenate([b-m, u], axis=1)) <= 0:
                # Compute the distance to the wall
                d = np.linalg.det(np.concatenate([a-m, b-a], axis=1))/np.linalg.det(np.concatenate([u, b-a], axis=1))
                
                # Keep only positive d
                if d > 0:
                    y[k] = np.min([y[k], d])
    return y

def pose_error(y, p, A, B):
    return np.linalg.norm(y - f(p, A, B))

def simulated_annealing_localization(p0, y, A, B, lamb=0.99):
    """Localization by simulated annealing

    Args:
        p0 (ndarray (3, 1)): initial guessed pose
        y (ndarray (8, 1)): laser measurements
        A (ndarray (2, n)): first extremity of walls
        B (ndarray (2, n)): second extremity of walls
        lamb (float, optional): Defaults to 0.99.

    Returns:
        _type_: _description_
    """
    # Init estimated pose
    p_estimated = np.copy(p0)
    
    # Initialize the error on the robot pose
    error = pose_error(y, p_estimated, A, B)
    
    # Initialize the temperature
    T = 10
    
    while T > 1 - lamb:
        # Realize a random motion from the current estimated pose
        p_new = p_estimated + T*np.random.randn(3, 1)
        
        # Compute the error corresponding to this new pose
        error_new = pose_error(y, p_new, A, B)
        
        # Compare this new error with the previous one
        if error_new < error:
            # If the new error is lower, update the estimation of the pose
            # of the robot
            p_estimated, error = p_new, error_new
        
        # Decrease the temperature
        T *= lamb
    
    return p_estimated


# Walls extremities
A = np.array([[0, 7, 7, 9, 9, 7, 7, 4, 2, 0, 5, 6, 6, 5],
              [0, 0, 2, 2, 4, 4, 7, 7, 5, 5, 2, 2, 3, 3]])
B = np.array([[7, 7, 9, 9, 7, 7, 4, 2, 0, 0, 6, 6, 5, 5],
              [0, 2, 2, 4, 4, 7, 7, 5, 5, 0, 2, 3, 3, 2]])

# Measured rays' length for an unknown pose of the robot
y = np.array([[6.4],[3.6],[2.3],[2.1],[1.7],[1.6],[3.0],[3.1]])                  

ax = init_figure(-2,10,-2,10)

# Set the pose of the robot
p = array([[1],[2],[3]])
draw_room()

# Compute laser rays' length
y_computed = f(p, A, B)

# Draw the robot and the rays
draw(p, y_computed, 'red')

# 10 seconds pause
pause(1)

# Localization by simulated annealing
# Initial guess of the robot pose (randomly chosen)
p0 = np.ones((3, 1))

# Localize the robot
p_estimated = simulated_annealing_localization(p0, y, A, B)

# Draw the robot
draw(p_estimated, f(p_estimated, A, B), 'red')
pause(3)