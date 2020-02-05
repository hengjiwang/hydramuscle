import numpy as np
from tqdm import tqdm


def euler_odeint(rhs, y, T, dt, save_interval=1, **kwargs):
    "An Euler-method integrator"

    sol = np.zeros((int(T/dt/save_interval), 80000))

    for j in tqdm(np.arange(0, int(T/dt))):
        if j % save_interval == 0:  
            sol[int(j/save_interval), 0:40000] = y[0:40000]
            sol[int(j/save_interval), 40000:80000] = y[14*40000:15*40000]
        t = j*dt
        dydt = rhs(y, t, **kwargs)
        y += dydt * dt

    return sol