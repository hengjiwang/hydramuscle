#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

c_m = 1e-6
g_cat = 0.0002
e_cat = 51
v0 = -50
A_cyt = 4e-5

T = 2
dt = 0.001
time = np.linspace(0, T, int(T/dt))

current = []

stages = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


def i_cat(v, bx, cx):
    return g_cat * bx**2 * cx * (v - e_cat)

def bx_inf(v):
    return 1 / (1 + np.exp(-(v+36.9)/6.6))

def cx_inf(v):
    return 1 / (1 + np.exp((v+63.8)/5.3))

def tau_bx(v):
    return (0.00045 + 0.0039 / (1 + ((v+66)/26)**2))

def tau_cx(v):
    return (0.15 - (0.15/((1+ np.exp((v-417.43)/203.18))*(1+np.exp(-(v+61.11)/8.07)))))

def stim(t):
    if t < 1.0: return 0
    else:   return 1

def rhs(y, t):
    
    v = -50 + stage * stim(t)
    
    bx, cx = y
    dbxdt = (bx_inf(v) - bx)/tau_bx(v)
    dcxdt = (cx_inf(v) - cx)/tau_cx(v)

    return [dbxdt, dcxdt]

def step():

    bx0 = bx_inf(v0)
    cx0 = cx_inf(v0)

    y0 = [bx0, cx0]
    sol = odeint(rhs, y0, time, hmax = 0.005)
    return sol

if __name__ == '__main__':

    minps = []

    plt.figure()
    for stage in stages:

        sol = step()
        bx = sol[:, 0]
        cx = sol[:, 1]

        stimulation = [stim(x) for x in time]
        vtg = np.array([-50 + stage * x for x in stimulation])
        crr = i_cat(vtg, bx, cx)
        minp = min(crr)
        minps.append(minp*A_cyt*1e9)
        
        # plt.plot(time, bx, 'b')
        # plt.plot(time, cx, 'r')
        plt.plot(time, crr*A_cyt*1e9)
    
    plt.show()

    plt.figure()
    plt.plot([-50+x for x in stages], minps)
    plt.show()