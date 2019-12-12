#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
from tqdm import tqdm

class FastCell:
    '''An intracellular model for calcium influx from extracellular space'''
    def __init__(self, T = 20, dt = 0.001):
        # General parameters
        self.c_m = 1e-6 # [F/cm^2]
        self.A_cyt = 4e-5 # [cm^2]
        self.V_cyt = 6e-9 # [cm^3]
        self.d = 20e-4 # [cm]
        self.F = 96485332.9 # [mA*s/mol]
        self.c0 = 0.05
        self.v0 = -50 # (-40 to -60)
        self.m0 = 0
        self.h0 = 0
        self.bx0 = 0
        self.cx0 = 0

        # Calcium leak
        self.tau_ex = 0.1 # [s]
        
        # CaL parameters
        self.g_cal = 0.0005 # [S/cm^2] 
        self.e_cal = 51
        self.k_cal = 1 # [uM]

        # CaT parameters
        self.g_cat = 0.003 # 0.0003
        self.e_cat = 51

        # BK parameters
        self.g_kca =  10e-9 / self.A_cyt # 45.7e-9 / self.A_cyt
        self.e_k = -75 

        # Background parameters
        self.g_bk = 0
        self.e_bk = -55

        # Time
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt))

    '''General functions'''
    def sig(self, v, vstar, sstar):
        # Sigmoidal function
        return 1 / (1 + np.exp(-(v-vstar)/sstar))

    def bell(self, v, vstar, sstar, taustar, tau0):
        # Bell-shape function
        return taustar/(np.exp(-(v-vstar)/sstar) + np.exp((v-vstar)/sstar)) + tau0

    '''CaL channel terms (Diderichsen 2006)'''
    def i_cal(self, v, m, h):
        # L-type calcium channel [mA/cm^2]
        return self.g_cal * m**2 * h * (v - self.e_cal)

    def m_inf(self, v):
        return self.sig(v, -25, 10)

    def h_inf(self, v):
        return self.sig(v, -28, -5)

    def tau_m(self, v):
        return self.bell(v, -23, 20, 0.001, 0.00005)

    def tau_h(self, v):
        return self.bell(v, 0, 20, 0.03, 0.021)

    '''T-type calcium channel terms (Mahapatra 2018)'''
    def i_cat(self, v, bx, cx):
        return self.g_cat * bx**2 * cx * (v - self.e_cat)

    def bx_inf(self, v):
        return self.sig(v, -32.1, 6.9)

    def cx_inf(self, v):
        return self.sig(v, -63.8, -5.3)

    def tau_bx(self, v):
        return 0.00045 + 0.0039 / (1 + ((v+66)/26)**2)

    def tau_cx(self, v):
        return 0.15 - 0.15 / ((1 + np.exp((v-417.43)/203.18))*(1 + np.exp(-(v+61.11)/8.07)))
    

    '''BK channel terms (Corrias 2007)'''
    def i_kca(self, v, c):
        return self.g_kca * 1 / (1 + np.exp(v/(-17) - 2 * np.log(c))) * (v - self.e_k)
        # return 5 * self.g_kca * c**2 / (c**2 + 5**2) * (v - self.e_k)

    '''Background terms'''
    def i_bk(self, v):
        # Background voltage leak [mA/cm^2]
        # g_bk = - (self.i_cal(self.v0, self.n0, self.hv0, self.hc0) \
        g_bk = - (self.i_cal(self.v0, self.m0, self.h0) \
        + self.i_cat(self.v0, self.bx0, self.cx0) \
        + self.i_kca(self.v0, self.c0))/(self.v0 - self.e_bk)

        return g_bk * (v - self.e_bk)

    '''Calcium terms'''
    def r_ex(self, c):
        # [uM/s]
        return (c-self.c0)/self.tau_ex

    '''Stimulation'''
    def stim_v(self, t, stims):

        condition = False

        for stim_t in stims:
            condition = condition or stim_t <= t < stim_t + 0.01

       	return int(condition)
           
    '''Numerical terms'''
    def rhs(self, y, t, stims):
        # Right-hand side function
        c, v, m, h, bx, cx = y
        dcdt = - self.r_ex(c) \
            - 1e9 * self.i_cal(v, m, h) / (2 * self.F * self.d) \
            + 1e9 * self.i_cal(self.v0, self.m0, self.h0) / (2 * self.F * self.d) \
            - 1e9 * self.i_cat(v, bx, cx) / (2 * self.F * self.d) \
            + 1e9 * self.i_cat(self.v0, self.bx0, self.cx0) / (2 * self.F * self.d)
        dvdt = - 1 / self.c_m * (self.i_cal(v, m, h) + self.i_cat(v, bx, cx) + self.i_kca(v, c) + self.i_bk(v) - 0.001 * self.stim_v(t, stims))
        dmdt = (self.m_inf(v) - m)/self.tau_m(v)
        dhdt = (self.h_inf(v) - h)/self.tau_h(v)
        dbxdt = (self.bx_inf(v) - bx)/self.tau_bx(v)
        dcxdt = (self.cx_inf(v) - cx)/self.tau_cx(v)

        return np.array([dcdt, dvdt, dmdt, dhdt, dbxdt, dcxdt])

    def step(self, stims = [1,3,5,7,9,11,13,15,17,19]):
        # Time stepping

        self.m0 = self.m_inf(self.v0)
        self.h0 = self.h_inf(self.v0)
        self.bx0 = self.bx_inf(self.v0)
        self.cx0 = self.cx_inf(self.v0)

        y0 = [self.c0, self.v0, self.m0, self.h0, self.bx0, self.cx0]

        start_time = time.time() # Begin counting time
        
        y = y0
        T = self.T
        dt = self.dt

        sol = np.zeros((int(T/dt/10), len(y0)))

        for j in tqdm(np.arange(0, int(T/dt))):
            t = j*dt
            dydt = self.rhs(y, t, stims)
            y += dydt * dt
            if j%10 == 0: sol[int(j/10), :] = y


        elapsed = (time.time() - start_time) # End counting time
        print("Time used:" + str(elapsed))

        return sol

    '''Visualize results'''
    def plot(self, a, tmin=0, tmax=100, xlabel = 'time[s]', ylabel = None, color = 'b'):
        x = np.linspace(tmin, tmax, len(a))
        plt.plot(x, a[int(tmin/self.dt):int(tmax/self.dt)], color)
        if xlabel:  plt.xlabel(xlabel)
        if ylabel:  plt.ylabel(ylabel)

if __name__ == '__main__':
    model = FastCell(10, 0.0002)
    sol = model.step()
    c = sol[:, 0]
    v = sol[:, 1]
    m = sol[:, 2]
    h = sol[:, 3]
    bx = sol[:, 4]
    cx = sol[:, 5]

    # Plot the results
    plt.figure()
    plt.subplot(421)
    model.plot(c, ylabel='c[uM]')
    plt.subplot(422)
    model.plot(v, ylabel='v[mV]')
    plt.subplot(423)
    model.plot(model.i_cal(v, m, h), ylabel='i_cal[mA/cm^2]')
    plt.subplot(424)
    model.plot(model.i_cat(v, bx, cx), ylabel='i_cat[mA/cm^2]')
    plt.subplot(425)
    model.plot(model.i_bk(v), ylabel='i_bk[mA/cm^2]')
    plt.subplot(426)
    model.plot(model.i_kca(v, c), ylabel = 'i_kca[mA/cm^2]')
    plt.subplot(427)
    model.plot([0.004 * model.stim_v(t, [1,3,5,7,9,11,13,15,17,19]) for t in model.time], ylabel='i_stim[mA/cm^2]', color = 'r')
    plt.show()