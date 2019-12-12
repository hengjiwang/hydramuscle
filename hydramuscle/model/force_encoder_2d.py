#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from force_encoder import ForceEncoder

class ForceEncoder2D(ForceEncoder):

    def __init__(self):
        