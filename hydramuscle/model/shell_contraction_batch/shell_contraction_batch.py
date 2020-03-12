import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time, random
import multiprocessing

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import spdiags
from tqdm import tqdm

from hydramuscle.model.smc import SMC
from hydramuscle.model.euler_odeint2 import euler_odeint

class Shell:

    def __init__(self, cell, numx, numy, save_interval):
        self.cell = cell
        self.T = cell.T
        self.dt = cell.dt
        self.gcx = 1000
        self.gcy = 1000
        self.gip3x = 10
        self.gip3y = 40
        self.numx = numx
        self.numy = numy
        self.num2 = self.numx*self.numy
        self.save_interval = save_interval

        self.v_scale = 0.01
        self.dur = 0.01

        self.init_connectivity_matrices()
        self.init_stimulation_pattern('fast', 0, int(self.numx/10), 0, self.numy, 0) # CB Stimulation


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

    def generate_indices(self, xmin, xmax, ymin, ymax):
        res = []

        for i in range(ymin, ymax):
            for j in range(xmin, xmax):
                res.append(i*self.numx+j)

        return res

    def init_stimulation_pattern(self, pathway, xmin, xmax, ymin, ymax, randomnum=0):

        if pathway == "fast":
            self.s_v = self.generate_indices(xmin, xmax, ymin, ymax)
            self.s_v += random.sample([j for j in range(self.num2)], randomnum)
        elif pathway == "slow":
            self.s_ip = self.generate_indices(xmin, xmax, ymin, ymax)
            self.s_ip += random.sample([j for j in range(self.num2)], randomnum)
        elif pathway == None:
            return
        else:
            print("Pathway wrong.")
            sys.exit(1)


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
            dipdt[self.s_ip] += self.cell.i_plcb(self.cell.stim_slow(t, stims_slow, self.cell.active_v8)) - self.cell.i_plcb(self.cell.v8)

        if self.s_v:
            dvdt[self.s_v] += 1 / self.cell.c_m * self.v_scale * self.cell.stim_fast(t, stims_fast, self.dur)

        deriv = np.array([dcdt, dsdt, drdt, dipdt, dvdt, dmdt, dhdt, dbxdt, dcxdt, dgdt, dc1gdt, dc2gdt, dc3gdt, dc4gdt])

        dydt = np.reshape(deriv, len(deriv)*num2)

        return dydt

    def run(self, stims_fast = [1,7,10,13,16,19,22,25,29,34,40,50], stims_slow = [-100]):
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
        sol = euler_odeint(self.rhs, y0, self.T, self.dt, 
                                    save_interval=self.save_interval, 
                                    numx=self.numx,
                                    numy=self.numy,
                                    layer_num=1,
                                    stims_fast=stims_fast, 
                                    stims_slow=stims_slow)

        return sol

if __name__ == "__main__":
    model = Shell(SMC(T=100, 
                      dt=0.0002, 
                      k2=float(sys.argv[1]), 
                      s0=int(sys.argv[2]), 
                      d=float(sys.argv[3]), 
                      v7=float(sys.argv[4]), 
                      active_v8=int(sys.argv[5])), 
                  numx=int(sys.argv[6]), 
                  numy=int(sys.argv[7]), 
                  save_interval=100)

    sol = model.run()
    df = pd.DataFrame(sol[:,0:model.numx*model.numy])
    df.to_csv('/media/hengji/DATA/Data/Documents/hydramuscle/results/data/calcium/'+
              str(model.numx) + 'x' + str(model.numy) + '_' +
              str(model.T) + 's_cb_' + 
              'k2_' + str(model.cell.k2).replace('.', '') +
              '_s0_' + str(model.cell.s0) + 
              '_v7_' + str(model.cell.v7).replace('.', '') + 
              '_activev8_' + str(model.cell.active_v8) + '.csv', index = False)