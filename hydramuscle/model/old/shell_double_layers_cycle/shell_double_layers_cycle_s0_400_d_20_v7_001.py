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
from hydramuscle.model.shell import Shell

class DoubleLayerShell:
    
    def __init__(self, numx=200, numy=200, save_interval=100):

        self.ectoderm = Shell(SMC(T=200, dt=0.0002, k2=0.2, s0=400, d=20e-4, v7=0.01), 
                              None, None, numx=numx, numy=numy, 
                              gip3x=10, gip3y=30, save_interval=None)
        self.endoderm = Shell(SMC(T=200, dt=0.0002, k2=0.2, s0=400, d=20e-4, v7=0.01), 
                              None, None, numx=numx, numy=numy, 
                              gip3x=10, gip3y=10, save_interval=None)

        self.ectoderm.init_stimulation_pattern('fast', 0, 10, 0, 200, 0) # CB Stimulation
        self.ectoderm.init_stimulation_pattern('slow', 0, 10, 90, 110, 0) # Bending Stimulation
        self.endoderm.init_stimulation_pattern('slow', 0, -1, 0, -1, 4000) # Active Elongation Stimulation


        self.T = self.ectoderm.T
        self.dt = self.ectoderm.dt
        self.gc = 500
        self.gip3 = 0
        self.save_interval = save_interval

        self.numx = numx
        self.numy = numy
        self.num2 = numx*numy

    def rhs(self, y, t, stims_fast_ecto, stims_slow_ecto, stims_fast_endo, stims_slow_endo):
        numx = self.numx
        numy = self.numy
        num2 = self.num2

        # Unpack dynamical variables
        c1, s1, r1, ip1, v1, m1, h1, bx1, cx1, g1, c1g1, c2g1, c3g1, c4g1 = (y[0:num2], 
        y[num2:2*num2], y[2*num2:3*num2], y[3*num2:4*num2], y[4*num2:5*num2], 
        y[5*num2:6*num2], y[6*num2:7*num2], y[7*num2:8*num2], y[8*num2:9*num2], 
        y[9*num2:10*num2], y[10*num2:11*num2], y[11*num2:12*num2], y[12*num2:13*num2],
        y[13*num2:14*num2])

        c2, s2, r2, ip2, v2, m2, h2, bx2, cx2, g2, c1g2, c2g2, c3g2, c4g2 = (y[14*num2:15*num2], 
        y[15*num2:16*num2], y[16*num2:17*num2], y[17*num2:18*num2], y[18*num2:19*num2], 
        y[19*num2:20*num2], y[20*num2:21*num2], y[21*num2:22*num2], y[22*num2:23*num2], 
        y[23*num2:24*num2], y[24*num2:25*num2], y[25*num2:26*num2], y[26*num2:27*num2],
        y[27*num2:28*num2])

        # Calculate terms
        i_ipr1, i_leak1, i_serca1, i_in1, i_pmca1, v_r1, i_plcd1, i_deg1 = self.ectoderm.cell.calc_slow_terms(c1, s1, r1, ip1)
        _, i_cal1, i_cat1, i_kca1, i_bk1, dmdt1, dhdt1, dbxdt1, dcxdt1 = self.ectoderm.cell.calc_fast_terms(c1, v1, m1, h1, bx1, cx1)
        ir11, ir21, ir31, ir41, dgdt1, dc1gdt1, dc2gdt1, dc3gdt1, dc4gdt1 = self.ectoderm.cell.calc_fluo_terms(c1, g1, c1g1, c2g1, c3g1, c4g1)

        i_ipr2, i_leak2, i_serca2, i_in2, i_pmca2, v_r2, i_plcd2, i_deg2 = self.endoderm.cell.calc_slow_terms(c2, s2, r2, ip2)
        _, i_cal2, i_cat2, i_kca2, i_bk2, dmdt2, dhdt2, dbxdt2, dcxdt2 = self.endoderm.cell.calc_fast_terms(c2, v2, m2, h2, bx2, cx2)
        ir12, ir22, ir32, ir42, dgdt2, dc1gdt2, dc2gdt2, dc3gdt2, dc4gdt2 = self.endoderm.cell.calc_fluo_terms(c2, g2, c1g2, c2g2, c3g2, c4g2)

        # Update dynamical variables
        dcdt1 = i_ipr1 + i_leak1 - i_serca1 + i_in1 - i_pmca1 - self.ectoderm.cell.alpha * (i_cal1 + i_cat1) - ir11 - ir21 - ir31 - ir41
        dsdt1 = self.ectoderm.cell.beta * (i_serca1 - i_ipr1 - i_leak1)
        drdt1 = v_r1
        dipdt1 = self.ectoderm.cell.i_plcb(self.ectoderm.cell.v8) + i_plcd1 - i_deg1 + self.ectoderm.Lip3.dot(ip1)
        dvdt1 = - 1 / self.ectoderm.cell.c_m * (i_cal1 + i_cat1 + i_kca1 + i_bk1) + self.ectoderm.Lc.dot(v1)

        dcdt2 = i_ipr2 + i_leak2 - i_serca2 + i_in2 - i_pmca2 - self.endoderm.cell.alpha * (i_cal2 + i_cat2) - ir12 - ir22 - ir32 - ir42
        dsdt2 = self.endoderm.cell.beta * (i_serca2 - i_ipr2 - i_leak2)
        drdt2 = v_r2
        dipdt2 = self.ectoderm.cell.i_plcb(self.endoderm.cell.v8) + i_plcd2 - i_deg2 + self.endoderm.Lip3.dot(ip2)
        dvdt2 = - 1 / self.endoderm.cell.c_m * (i_cal2 + i_cat2 + i_kca2 + i_bk2) + self.endoderm.Lc.dot(v2)

        # Ecto-endo communication
        dvdt1 += self.gc*(v2-v1)
        dvdt2 += self.gc*(v1-v2)
        dipdt1 += self.gip3*(ip2-ip1)
        dipdt2 += self.gip3*(ip1-ip2)

        # Add stimulation
        if self.ectoderm.s_ip:
            print()
            dipdt1[self.ectoderm.s_ip] += self.ectoderm.cell.i_plcb(self.ectoderm.cell.stim_slow(t, stims_slow_ecto)) - self.ectoderm.cell.i_plcb(self.ectoderm.cell.v8)

        if self.ectoderm.s_v:
            dvdt1[self.ectoderm.s_v] += 1 / self.ectoderm.cell.c_m * self.ectoderm.v_scale * self.ectoderm.cell.stim_fast(t, stims_fast_ecto, self.ectoderm.dur)

        if self.endoderm.s_ip:
            dipdt2[self.endoderm.s_ip] += self.endoderm.cell.i_plcb(self.endoderm.cell.stim_slow(t, stims_slow_endo)) - self.endoderm.cell.i_plcb(self.endoderm.cell.v8)

        if self.endoderm.s_v:
            dvdt2[self.endoderm.s_v] += 1 / self.endoderm.cell.c_m * self.endoderm.v_scale * self.endoderm.cell.stim_fast(t, stims_fast_endo, self.endoderm.dur)

        deriv = np.array([dcdt1, dsdt1, drdt1, dipdt1, dvdt1, dmdt1, dhdt1, dbxdt1, dcxdt1, dgdt1, dc1gdt1, dc2gdt1, dc3gdt1, dc4gdt1,
                          dcdt2, dsdt2, drdt2, dipdt2, dvdt2, dmdt2, dhdt2, dbxdt2, dcxdt2, dgdt2, dc1gdt2, dc2gdt2, dc3gdt2, dc4gdt2])

        dydt = np.reshape(deriv, len(deriv)*num2)

        return dydt

    def run(self, stims_fast_ecto=[1,5,9,13,17,21,25,30,35,40,46,53,61,70], stims_slow_ecto=[80], stims_fast_endo=[], stims_slow_endo=[90]):

        self.ectoderm.cell.init_fast_cell()
        self.ectoderm.cell.init_slow_cell()
        self.endoderm.cell.init_fast_cell()
        self.endoderm.cell.init_slow_cell()

        base_mat = np.ones((self.numy, self.numx))

        inits = [self.ectoderm.cell.c0, 
                 self.ectoderm.cell.s0, 
                 self.ectoderm.cell.r0, 
                 self.ectoderm.cell.ip0, 
                 self.ectoderm.cell.v0, 
                 self.ectoderm.cell.m0, 
                 self.ectoderm.cell.h0,
                 self.ectoderm.cell.bx0, 
                 self.ectoderm.cell.cx0, 
                 self.ectoderm.cell.fluo_buffer.g0, 
                 self.ectoderm.cell.fluo_buffer.c1g0, 
                 self.ectoderm.cell.fluo_buffer.c2g0, 
                 self.ectoderm.cell.fluo_buffer.c3g0, 
                 self.ectoderm.cell.fluo_buffer.c4g0,
                 self.endoderm.cell.c0, 
                 self.endoderm.cell.s0, 
                 self.endoderm.cell.r0, 
                 self.endoderm.cell.ip0, 
                 self.endoderm.cell.v0, 
                 self.endoderm.cell.m0, 
                 self.endoderm.cell.h0,
                 self.endoderm.cell.bx0, 
                 self.endoderm.cell.cx0, 
                 self.endoderm.cell.fluo_buffer.g0, 
                 self.endoderm.cell.fluo_buffer.c1g0, 
                 self.endoderm.cell.fluo_buffer.c2g0, 
                 self.endoderm.cell.fluo_buffer.c3g0, 
                 self.endoderm.cell.fluo_buffer.c4g0]

        y0 = np.array([x*base_mat for x in inits])
        y0 = np.reshape(y0, len(inits)*self.num2)  

        # Begin counting time
        sol = euler_odeint(self.rhs, y0, self.T, self.dt, 
                                    numx=200,
                                    numy=200,
                                    layer_num=2,
                                    save_interval=self.save_interval, 
                                    stims_fast_ecto=stims_fast_ecto, 
                                    stims_slow_ecto=stims_slow_ecto,
                                    stims_fast_endo=stims_fast_endo, 
                                    stims_slow_endo=stims_slow_endo)

        return sol

if __name__ == "__main__":
    model = DoubleLayerShell()
    sol = model.run()
    df1 = pd.DataFrame(sol[:,0:model.num2])
    df2 = pd.DataFrame(sol[:,model.num2:2*model.num2])
    df1.to_csv('/media/hengji/DATA/Data/Documents/hydramuscle/results/data/calcium/200x200_200s_cycle_ecto_s0_400_d_20_v7_001.csv', index = False)
    time.sleep(600)
    df2.to_csv('/media/hengji/DATA/Data/Documents/hydramuscle/results/data/calcium/200x200_200s_cycle_endo_s0_400_d_20_v7_001.csv', index = False)