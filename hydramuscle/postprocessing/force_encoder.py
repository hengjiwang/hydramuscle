import numpy as np
from scipy.integrate import odeint

class ForceEncoder(object):

    # Attachment & Detachment rates
    k2 = 0.1399
    k3 = 14.4496
    k4 = 3.6124
    k5 = k2
    k6 = 0
    k7 = 0.1340

    # General parameters
    nm = 4.7135
    c_half = 0.4640758 # 1
    K = 5.0859 # 1 

    # Initial variables
    m0 = 1
    mp0 = 0
    amp0 = 0
    am0 = 0

    @classmethod
    def _rhs(cls, y, c):
        # Right-hand side formulation

        k1 = c**cls.nm / (c**cls.nm + cls.c_half**cls.nm)
        cls.k6 = k1

        trans = np.array([[-k1, cls.k2, 0, cls.k7], 
                          [k1, -cls.k2 - cls.k3, cls.k4, 0],
                          [0, cls.k3, -cls.k4-cls.k5, cls.k6],
                          [0, 0, cls.k5, - cls.k6 - cls.k7]])

        return list(trans@np.array(y))

    @classmethod
    def encode(cls, c, dt):
        # Encode c to active force

        cls.dt = dt
        cls.T = len(c)*cls.dt
        cls.time = np.linspace(0, cls.T, int(cls.T/cls.dt)+1)

        y0 = [cls.m0, cls.mp0, cls.amp0, cls.am0]
        sol = odeint(cls._rhs, y0, cls.time, hmax = 0.005, args=(c,))
        return cls.K * (sol[:,2] + sol[:,3])

    
