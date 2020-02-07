import numpy as np
from tqdm import tqdm


def euler_odeint(rhs, y, T, dt, save_interval=1, numx=200, numy=200, layer_num=2, **kwargs):
    "An Euler-method integrator"

    sol = np.zeros((int(T/dt/save_interval), layer_num*numx*numy))

    for j in tqdm(np.arange(0, int(T/dt))):
        if j % save_interval == 0:  
            sol[int(j/save_interval), 0:numx*numy] = y[0:numx*numy]
            if layer_num == 2:  sol[int(j/save_interval), numx*numy:2*numx*numy] = y[14*numx*numy:15*numx*numy]
        t = j*dt
        dydt = rhs(y, t, **kwargs)
        y += dydt * dt

    return sol