import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time, random

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from tqdm import tqdm

from hydra.model.smc import SMC
from hydra.model.euler_odeint2 import euler_odeint
from hydra.model.layer import Layer

class Shell:
    """ Two-layer body shell """

    def __init__(self, ectoderm, endoderm, sparsity=0.1, gc=1000, gip3=5):

        self.ectoderm = ectoderm
        self.endoderm = endoderm
        
        self._numx = self.ectoderm.numx
        self._numy = self.ectoderm.numy
        self._num2 = self._numx * self._numy

        # Numerical settings (follow layer model)
        self.T = self.ectoderm.T
        self.dt = self.endoderm.dt
        self._save_interval = self.ectoderm.save_interval

        # Init cross-layer gap junctions
        self.cross_layer_pattern = np.random.uniform(0, 1, size=self._num2)
        self.cross_layer_pattern = (self.cross_layer_pattern < sparsity).astype(float)
        self.gc = self.cross_layer_pattern * gc
        self.gip3 = self.cross_layer_pattern * gip3

    def _rhs(self, y, t):
        "Right-hand side"

        # Calculate derivatives in the two layers
        dc1dt, ds1dt, dr1dt, dip1dt, dv1dt, dm1dt, dh1dt, dn1dt = self.ectoderm.calc_derivs(y[0:8*self._num2], t)
        dc2dt, ds2dt, dr2dt, dip2dt, dv2dt, dm2dt, dh2dt, dn2dt = self.endoderm.calc_derivs(y[8*self._num2:16*self._num2], t)

        # Correct the dipdt and dvdt with cross-layer coupling
        v1, v2 = y[4*self._num2:5*self._num2], y[12*self._num2:13*self._num2]
        ip1, ip2 = y[3*self._num2:4*self._num2], y[11*self._num2:12*self._num2]
        dip1dt += self.gip3 * (ip2 - ip1)
        dip2dt += self.gip3 * (ip1 - ip2)
        dv1dt += self.gip3 * (v2 - v1)
        dv2dt += self.gip3 * (v1 - v2)

        # Re-pack the derivatives
        deriv = np.array([dc1dt, ds1dt, dr1dt, dip1dt, dv1dt, dm1dt, dh1dt, dn1dt,
                          dc2dt, ds2dt, dr2dt, dip2dt, dv2dt, dm2dt, dh2dt, dn2dt])

        dydt = np.reshape(deriv, len(deriv)*self._num2)

        return dydt

    def run(self):
        "Run the model"
        self.ectoderm.cell.init_fast_cell()
        self.ectoderm.cell.init_slow_cell()
        self.endoderm.cell.init_fast_cell()
        self.endoderm.cell.init_slow_cell()

        base_mat = np.ones((self._numy, self._numx))

        inits = [self.ectoderm.cell.c0,
                 self.ectoderm.cell.s0,
                 self.ectoderm.cell.r0,
                 self.ectoderm.cell.ip0,
                 self.ectoderm.cell.v0,
                 self.ectoderm.cell.m0,
                 self.ectoderm.cell.h0,
                 self.ectoderm.cell.n0,
                 self.endoderm.cell.c0,
                 self.endoderm.cell.s0,
                 self.endoderm.cell.r0,
                 self.endoderm.cell.ip0,
                 self.endoderm.cell.v0,
                 self.endoderm.cell.m0,
                 self.endoderm.cell.h0,
                 self.endoderm.cell.n0]

        y0 = np.array([x*base_mat for x in inits])
        y0 = np.reshape(y0, len(inits)*self._num2)  

        # Begin counting time
        sol_ = euler_odeint(rhs=self._rhs,
                            y=y0,
                            T=self.T,
                            dt=self.dt,
                            save_interval=self._save_interval,
                            numx=self._numx,
                            numy=self._numy,
                            layer_num=2)

        return sol_

if __name__ == "__main__":
    pass