import numpy as np
from tqdm import tqdm


def euler_odeint(rhs, y, T, dt, save_interval=1, **kwargs):
    "An Euler-method integrator"

    sol = np.zeros((int(T/dt/save_interval), len(y)))

    for j in tqdm(np.arange(0, int(T/dt))):
        if j % save_interval == 0:
            sol[int(j/save_interval), :] = y
        t = j*dt
        dydt = rhs(y, t, **kwargs)
        y += dydt * dt

    return sol