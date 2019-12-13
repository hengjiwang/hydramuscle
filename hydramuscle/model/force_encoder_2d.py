import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from hydramuscle.model.force_encoder import ForceEncoder

class ForceEncoder2D(ForceEncoder):

    def __init__(self):
        pass