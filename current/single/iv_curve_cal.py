#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

c_m = 1e-6
g_cal = 0.0004
e_cal = 51
k_cal = 1
v0 = -50
A_cyt = 4e-5
c = 1

T = 2
dt = 0.001
time = np.linspace(0, T, int(T/dt))

current = []

stages = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


def i_cal(v, n, hv, hc):
    return g_cal * n**2 * hv * hc * (v - e_cal)

def n_inf(v):
    return 1 / (1 + np.exp(-(v+9)/8))

def hv_inf(v):
    return 1 / (1 + np.exp((v+30)/13))

def hc_inf(c):
    return k_cal / (k_cal + c)

def tau_n(v):
    return 0.000001 / (1 + np.exp(-(v+22)/308))

def tau_hv(v):
    return 0.09 * (1 - 1 / ((1 + np.exp((v+14)/45)) * (1 + np.exp(-(v+9.8)/3.39))))

def tau_hc():
    return 0.02

def stim(t):
    if t < 1.0: return 0
    else:   return 1

def rhs(y, t):
    
    v = -50 + stage * stim(t)
    
    n, hv, hc = y
    dndt = (n_inf(v) - n)/tau_n(v)
    dhvdt = (hv_inf(v) - hv)/tau_hv(v)
    dhcdt = (hc_inf(c) - hc)/tau_hc()
    
    return [dndt, dhvdt, dhcdt]

def step():

    n0 = n_inf(v0)
    hv0 = hv_inf(v0)
    hc0 = hc_inf(c)

    y0 = [n0, hv0, hc0]
    sol = odeint(rhs, y0, time, hmax = 0.005)
    return sol

if __name__ == '__main__':

    minps = []

    plt.figure()
    for stage in stages:

        sol = step()
        n = sol[:, 0]
        hv = sol[:, 1]
        hc = sol[:, 2]

        stimulation = [stim(x) for x in time]
        vtg = np.array([-50 + stage * x for x in stimulation])
        crr = i_cal(vtg, n, hv, hc)
        minp = min(crr)
        minps.append(minp*A_cyt*1e9)
        
        # plt.plot(time, bx, 'b')
        # plt.plot(time, cx, 'r')
        plt.plot(time, crr*A_cyt*1e9)
    
    plt.show()

    plt.figure()
    plt.plot([-50+x for x in stages], minps, 'ro-')
    plt.show()