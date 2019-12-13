import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time, random

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import spdiags
from tqdm import tqdm

from hydramuscle.model.smc import SMC

class Shell:

    def __init__(self, cell, behavior, numx=200, numy=200):
        self.cell = cell
        self.T = cell.T
        self.dt = cell.dt
        self.gcx = 1000
        self.gcy = 1000
        self.gip3x = 2
        self.gip3y = 2
        self.numx = numx
        self.numy = numy
        self.num2 = self.numx*self.numy
        self.behavior = behavior
        self.init_connectivity_matrices()
        self.init_stimulation_pattern(behavior)


    def init_connectivity_matrices(self):
        onex = np.ones(self.numx)
        Ix = np.eye(self.numx)
        oney = np.ones(self.numy)
        Iy = np.eye(self.numy)
        Dx = spdiags(np.array([onex,-2*onex,onex]),np.array([-1,0,1]),self.numx,self.numx).toarray()
        Dy = spdiags(np.array([oney,-2*oney,oney]),np.array([-1,0,1]),self.numy,self.numy).toarray()
        Dx[0, self.numx-1] = 1
        Dx[self.numx-1, 0] = 1
        Dy[0,0] = -1
        Dy[self.numy-1,self.numy-1] = -1 
        Dx = scipy.sparse.csr_matrix(Dx)
        Dy = scipy.sparse.csr_matrix(Dy)
        Ix = scipy.sparse.csr_matrix(Ix)
        Iy = scipy.sparse.csr_matrix(Iy)
        self.s_ip = None
        self.s_v = None
        self.Lc = self.gcx * scipy.sparse.kron(Dx, Iy) + self.gcy * scipy.sparse.kron(Ix, Dy)
        self.Lip3 = self.gip3x * scipy.sparse.kron(Dx, Iy) + self.gip3y * scipy.sparse.kron(Ix, Dy)

    def init_stimulation_pattern(self, behavior):
        if behavior == 'contraction burst':
            self.s_v = [self.numy*i for i in range(self.numx)]
        elif behavior == 'elongation':
            self.s_ip = [j for j in range(self.num2)]
            self.s_ip = random.sample(self.s_ip, 4000)
        elif behavior == 'bending':
            self.s_ip = [(int(self.numx/2)-j)*self.numy for j in range(-20, 20)]

    def rhs(self, y, t, stims_fast, stims_slow):
        # Right-hand side equations

        numx = self.numx
        numy = self.numy
        num2 = self.num2

        # Unpack dynamical variables
        c, s, r, ip, v, m, h, bx, cx, g, c1g, c2g, c3g, c4g = (y[0:num2], 
        y[num2:2*num2], y[2*num2:3*num2], y[3*num2:4*num2], y[4*num2:5*num2], 
        y[5*num2:6*num2], y[6*num2:7*num2], y[7*num2:8*num2], y[8*num2:9*num2], 
        y[9*num2:10*num2], y[10*num2:11*num2], y[11*num2:12*num2], y[12*num2:13*num2],
        y[13*num2:14*num2])

        # Calculate terms
        i_ipr, i_leak, i_serca, i_in, i_pmca, v_r, i_plcd, i_deg = self.cell.calc_slow_terms(c, s, r, ip)
        _, i_cal, i_cat, i_kca, i_bk, dmdt, dhdt, dbxdt, dcxdt = self.cell.calc_fast_terms(c, v, m, h, bx, cx)
        ir1, ir2, ir3, ir4, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt = self.cell.calc_fluo_terms(c, g, c1g, c2g, c3g, c4g)

        # Update dynamical variables
        dcdt = i_ipr + i_leak - i_serca + i_in - i_pmca - self.cell.alpha * (i_cal + i_cat) - ir1 - ir2 - ir3 - ir4
        dsdt = self.cell.beta * (i_serca - i_ipr - i_leak)
        drdt = v_r
        dipdt = self.cell.i_plcb(self.cell.v8) + i_plcd - i_deg + self.Lip3.dot(ip)
        dvdt = - 1 / self.cell.c_m * (i_cal + i_cat + i_kca + i_bk) + self.Lc.dot(v)

        # Add stimulation
        if self.s_ip:
            dipdt[self.s_ip] += self.cell.i_plcb(self.cell.stim_slow(t, stims_slow)) - self.cell.i_plcb(self.cell.v8)

        if self.s_v:
            dvdt[self.s_v] += 1 / self.cell.c_m * 0.01 * self.cell.stim_fast(t, stims_fast)

        deriv = np.array([dcdt, dsdt, drdt, dipdt, dvdt, dmdt, dhdt, dbxdt, dcxdt, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt])

        dydt = np.reshape(deriv, len(deriv)*num2)

        return dydt

    def run(self, stims_fast = [101,103,105,107,109,112,115,118,122,126,131,136,142,148], stims_slow = [-100]):
        # Time stepping
        
        self.cell.init_fast_cell()
        self.cell.init_slow_cell()

        base_mat = np.ones((self.numy, self.numx))

        inits = [self.cell.c0, 
                 self.cell.s0, 
                 self.cell.r0, 
                 self.cell.ip0, 
                 self.cell.v0, 
                 self.cell.m0, 
                 self.cell.h0,
                 self.cell.bx0, 
                 self.cell.cx0, 
                 self.cell.fluo_buffer.g0, 
                 self.cell.fluo_buffer.c1g0, 
                 self.cell.fluo_buffer.c2g0, 
                 self.cell.fluo_buffer.c3g0, 
                 self.cell.fluo_buffer.c4g0]

        y0 = np.array([x*base_mat for x in inits])
        y0 = np.reshape(y0, len(inits)*self.num2)  

        # Begin counting time
        sol = self.cell.euler_odeint(self.rhs, y0, self.T, self.dt, 
                                    save_interval=200, 
                                    stims_fast=stims_fast, 
                                    stims_slow=stims_slow)

        return sol

if __name__ == "__main__":
    model = Shell(SMC(T=100, dt=0.0002, k2=0.1, s0=400, d=40e-4, v7=0.01), 
    'contraction burst', numx=200, numy=200)
    sol = model.run([1,3,5,7,9,12,15,18,22,26,31,36,42])
    df = pd.DataFrame(sol[:,0:model.numx*model.numy])
    df.to_csv('~/Documents/hydra_calcium_model/save/data/calcium/c_200x200_100s_cb_refactored.csv', index = False)
