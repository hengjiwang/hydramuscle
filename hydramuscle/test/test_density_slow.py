import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys

from hydramuscle.model.smc import SMC
from hydramuscle.model.layer import Layer
from hydramuscle.model.shell import Shell
from hydramuscle.model.fluo_encoder import FluoEncoder
from hydramuscle.utils import utils

if __name__ == "__main__":
    smc = SMC(k_ipr=0.2, k_deg=0.05, s0=100, d=20e-4, T=30, dt=0.0002)
    ectoderm = Layer(smc, numx=30, numy=60, gip3x=0.1, gip3y=2, gcx=1000, gcy=1000, save_interval=100, active_v_beta=1)
    endoderm = Layer(smc, numx=30, numy=60, gip3x=0.1, gip3y=2, gcx=1000, gcy=1000, save_interval=100, active_v_beta=1)

    stims_slow = [0]
    ectoderm.set_stim_pattern("slow", xmin=13, xmax=17, ymin=0, ymax=4, stim_times=stims_slow)

    # Sweep sparsity
    for i in range(20):
        shell = Shell(ectoderm, endoderm, seed=np.random.randint(0, 10000), sparsity=float(sys.argv[1]), gc=1000, gip3=0.1)
        ca = shell.run()
        ca = pd.DataFrame(ca)
        ca.to_hdf("../results/data/calcium/cross_layer_density_slow/density_" + sys.argv[1] + "_" + str(i) + ".h5", 'calcium')