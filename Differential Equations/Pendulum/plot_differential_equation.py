''' Pendulum Differential Equation

    theta_ddot + b/m * theta_dot + g/L * sin(theta) = 0
    
    To solve it, convert it into two ODE of order 1:

    theta = theta1
    theta1_dot = theta2

    theta2_dot = -b/m * theta1_dot - g/L sin(theta1)

    https://projects.skill-lync.com/projects/Solving-2nd-order-ODE-for-a-simple-pendulum-using-python-40080
'''

import numpy as np
from scipy.integrate import odeint
import math
import matplotlib.pyplot as plt

# Physical constants
b = 0.01  # damping coefficient
g = 9.81  # m/s^2
L = 1     # m
m = 1     # kg

# Differential Equation
def system(state, b, g, l, m):
    x, y = state  # angular displacement, angular velocity

    dy = (-(b/m) * y - (g/l) * np.sin(x)) * np.power(np.sign(y), 2)
    dx = y * np.power(np.sign(y), 2)

    return dx, dy


max_value = 20
intervals = 1

x = np.arange(-max_value, max_value, intervals)
y = np.arange(-max_value, max_value, intervals)
x = np.arange(-5, 5, .2)
y = np.arange(-2, 2, .1)

X, Y = np.meshgrid(x, y)

dx, dy = system([X, Y], b, g, L, m)

lengths = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
dx_normed = dx / lengths
dy_normed = dy / lengths

fig, ax = plt.subplots(figsize=(12,12))

#ax.quiver(X, Y, dx_normed, dy_normed)
ax.quiver(X, Y, dx, dy)

#ax.xaxis.set_ticks([])
#ax.yaxis.set_ticks([])
#ax.set_aspect('equal')

plt.show()