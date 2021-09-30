import numpy as np
import math

def f_numpy_wrap(f):
    def new_func(x,t):
        return np.array(f(x,t))
    return new_func

def rk_step(f, x_start, t, dt):
    k_1 = f(x_start, t)
    k_2 = f(x_start + (dt/2)*k_1, t + dt/2)
    k_3 = f(x_start + (dt/2)*k_2, t + dt/2)
    k_4 = f(x_start + dt*k_3, t + dt)
    # print("k_1 {}, k_2 {}, k_3 {}, k_4 {}".format(k_1,k_2,k_3,k_4))

    x_new = x_start + (dt/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
    return x_new

def runge_kutta(f, x_0, dt, runtime, t_0 = 0):
    f = f_numpy_wrap(f)
    x_0 = np.array(x_0)
    x_current = x_0
    t_current = t_0

    steps = math.floor(runtime / dt)
    # Array of all x values, 0, 1, ...,
    x = np.zeros((steps + 1, len(x_0)))
    t = np.zeros((steps+1,))
    x[0] = x_0

    if steps < 5:
        raise Exception("With runtime={}, dt={}, number of steps is just {}. is that right?".format(runtime,dt, steps))
    for n in range(1, steps + 1):
        # print(x_current)
        x_new = rk_step(f, x_current, t_current, dt)
        t_new = t_current + dt
        # print("current: {}, new: {}".format(x_current,x_new))
        x[n] = x_new
        t[n] = t_new
        x_current = x_new
        t_current = t_new

    return x,t


