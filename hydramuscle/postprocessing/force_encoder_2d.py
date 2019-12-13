import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

from hydramuscle.postprocessing.force_encoder import ForceEncoder
from hydramuscle.lib.euler_odeint import euler_odeint

class ForceEncoder2D(ForceEncoder):

    @classmethod
    def _rhs(cls, y, c, num2):
        k1 = c**cls.nm / (c**cls.nm + cls.c_half**cls.nm)
        cls.k6 = k1

        m, mp, amp, am = y[0:num2], y[num2:2*num2], y[2*num2:3*num2], y[3*num2:4*num2]

        dmdt = -k1*m + cls.k2*mp + cls.k7*am
        dmpdt = k1*m + (-cls.k2 - cls.k3)*mp + cls.k4*amp
        dampdt = cls.k3*mp + (-cls.k4-cls.k5)*amp + cls.k6*am
        damdt = cls.k5*amp + (-cls.k6-cls.k7)*am

        dydt = np.reshape([dmdt, dmpdt, dampdt, damdt], 4*num2)

        return dydt

    @classmethod
    def encode(cls, c, numx, numy, dt):
        "Encode calcium into active force"

        cls.dt = dt
        cls.T = len(c)*cls.dt
        cls.time = np.linspace(0, cls.T, int(cls.T/cls.dt)+1)

        num2 = numx*numy
        base_mat = np.ones((numy, numx))
        inits = [cls.m0, cls.mp0, cls.amp0, cls.am0]
        y0 = np.array([x*base_mat for x in inits])
        y0 = np.reshape(y0, 4*num2) 
        
        sol = euler_odeint(cls._rhs, y0, cls.T, cls.dt, num2=num2)


        return cls.K * (sol[:,2*num2:3*num2] + sol[:,3*num2:4*num2])
        
if __name__ == "__main__":
    c = pd.read_csv("../save/data/calcium/c_200x200_100s_elong.csv").values
    force = ForceEncoder2D.encode(c, 200, 200, 0.02)
    df = pd.DataFrame(force)
    df.to_csv('../../save/data/force/force_200x200_100s_elong.csv', index = False)