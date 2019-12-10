#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class CellBase(object):
    '''A base class for simulating single cell dynamics'''
    def __init__(T, dt):
        self.T = T
        self.dt = dt
        self.time = np.linspace(0, T, int(T/dt)+1)