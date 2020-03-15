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
b = 0.2  # damping coefficient
g = 9.81  # m/s^2
L = 4     # m
m = 1     # kg

# Initial condition
theta_0 = [0, 3]

# Time span
t = np.linspace(0, 20, 240)  # 20s with 240 intervals

# Differential Equation
def system(theta, t, b, g, l, m):
    theta1 = theta[0]  # angular displacement
    theta2 = theta[1]  # angular velocity

    theta1_dot = theta2
    theta2_dot = -(b/m) * theta2 - g * math.sin(theta1)
    
    theta_dot = [theta1_dot, theta2_dot]
    return theta_dot

theta = odeint(system, theta_0, t, args=(b, g, L, m))

# Animation
folder = 'build/'
f = 1
import progressbar
for i in progressbar.progressbar(range(240)):
    filename = folder + str(f) + '.png'
    f += 1
    fig = plt.figure()

    # (10, 10) is the start point of the Pendulum
    plt.plot([10, L * math.sin(theta[i, 0]) + 10], [10, 10 - L * math.cos(theta[i, 0])], marker='o')
    plt.xlim([0, 20])
    plt.ylim([0, 20])

    plt.savefig(filename)
    plt.close(fig)

plt.plot(t, theta[:,0], 'b-')  # plot angular displacement over time
plt.plot(t, theta[:,1], 'r--')  # plot angular velocity over time
plt.show()


