import numpy as np
import matplotlib.pyplot as plt
import math
from utils import runge_kutta


def pendulum_func(state, t):
    phi, phi_dot = state

    phi_derivative = phi_dot
    phi_dot_derivative = - 4*math.pi*math.pi*math.sin(phi)

    return phi_derivative, phi_dot_derivative

if __name__ == "__main__":

    init_angle = 0
    for init_velocity in [2, 8, 12.5, 12.6, 12.8, 14]:
        x,t = runge_kutta(pendulum_func, (init_angle, init_velocity), 0.0002, 3)
        angles = x[:,0]
        angles = np.mod(angles + np.pi, 2*np.pi) - np.pi
        horizontal = np.zeros_like(angles)
        print(x)
        print(t)
        # print(x)
        # plt.close()
        plt.plot(t,angles,label=r'$\dot{\phi}(0)=' + str(init_velocity) + "$")
        # if init_vel
        # plt.plot(t,horizontal)
    plt.legend()
    plt.show()
