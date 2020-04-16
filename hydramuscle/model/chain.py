import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from hydramuscle.model.smc import SMC
from hydramuscle.model.euler_odeint import euler_odeint

class Chain:

    def __init__(self, cell, )