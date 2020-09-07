import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# from hydra.model.euler_odeint import euler_odeint

class ForceEncoderEcto(object):

    # Attachment & Detachment rates
    k2 = 0.15 # 0.15 # 0.1399
    k3 = 16 # 0.4 # 14.4496
    k4 = 4 # 0.05 # 3.6124
    k5 = k2
    k6 = 0
    k7 = 0.75 # 0.15 # 0.07 # 0.1 # 0.05 # 0.1340

    # General parameters
    nm = 4 # 4.7135
    c_half = 0.85 # 0.5 # 0.4640758 # 1
    K = 2 # 2.6 # 4

    # Initial variables
    m0 = 1
    mp0 = 0
    amp0 = 0
    am0 = 0

    @classmethod
    def _rhs(cls, y, t, calcium):
        # Right-hand side formulation

        if t < cls.T: c = calcium[int(t/cls.dt)] # [0]
        else: c = calcium[-1] # [0]

        k1 = c**cls.nm / (c**cls.nm + cls.c_half**cls.nm)
        cls.k6 = k1

        trans = np.array([[-k1, cls.k2, 0, cls.k7], 
                          [k1, -cls.k2 - cls.k3, cls.k4, 0],
                          [0, cls.k3, -cls.k4-cls.k5, cls.k6],
                          [0, 0, cls.k5, - cls.k6 - cls.k7]])


        return list(trans@np.array(y))

    @classmethod
    def encode(cls, calcium, dt):
        # Encode c to active force

        cls.dt = dt
        cls.T = (len(calcium)-1)*cls.dt
        cls.time = np.linspace(0, cls.T, int(cls.T/cls.dt)+1)

        y0 = [cls.m0, cls.mp0, cls.amp0, cls.am0]
        sol = odeint(cls._rhs, y0, cls.time, args=(calcium,), hmax=0.005)
        return cls.K * (sol[:, 2] + sol[:, 3])

    
if __name__ == "__main__":
    calcium = pd.read_csv('../save/data/calcium/c_sin_ibk.csv').values
    force = ForceEncoderEcto.encode(calcium, 0.0002)

    plt.figure()
    plt.plot(np.linspace(0, 200, int(200/0.0002)+1), force)
    plt.show()