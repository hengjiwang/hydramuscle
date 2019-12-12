#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class FluoBuffer(object):
    
    # Init
    r_inc = 200
    tau_ex = 1.0
    k1p = 2.5
    k1n = 0.1
    k2p = 16.9
    k2n = 205
    k3p = 1.1
    k3n = 11.8
    k4p = 1069
    k4n = 5.8
    g0 = 3.28088468
    c1g0 = 4.10110585
    c2g0 = 0.01690456
    c3g0 = 7.87924326e-05
    c4g0 = 0.00072611
    phi0 = 1
    phi1 = 1
    phi2 = 1
    phi3 = 1
    phi4 = 81
        
    
    # Fluorescence
    @classmethod
    def f_total(cls, g, c1g, c2g, c3g, c4g):
        f_cyt = cls.phi0*g + cls.phi1*c1g + cls.phi2*c2g + \
        cls.phi3*c3g + cls.phi4*c4g
        f_cyt0 = cls.phi0*cls.g0 + cls.phi1*cls.c1g0 + \
        cls.phi2*cls.c2g0 + cls.phi3*cls.c3g0 + cls.phi4*cls.c4g0
        f_bg = f_cyt0 * 2.5
        return (f_cyt + f_bg) / (f_cyt0 + f_bg)
    
    # rate 1
    @classmethod
    def r_1(cls, c, g, c1g):
        return cls.k1p * c * g - cls.k1n * c1g
    
    # rate 2
    @classmethod
    def r_2(cls, c, c1g, c2g):
        return cls.k2p * c * c1g - cls.k2n * c2g
    
    # rate_3
    @classmethod
    def r_3(cls, c, c2g, c3g):
        return cls.k3p * c * c2g - cls.k3n * c3g
    
    # rate_4
    @classmethod
    def r_4(cls, c, c3g, c4g):
        return cls.k4p * c * c3g - cls.k4n * c4g
    