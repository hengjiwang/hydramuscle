import numpy as np
import matplotlib.pyplot as plt

class ReducedFastCell:

    def __init__(self, dt=0.001):
        self.dt = dt
        self.t = 0
        self.c0 = 0.05 # uM
        self.tau_stim = 0.05 # s
        self.r_stim = 1 # uM/s
        self.tau_inc = 0.075 # uM/s
        self.tau_dec = 0.5 # s
        self.tau_ref = 0.1 # s
        self.last_stim = -100 # s
        self.c = self.c0
        self.c_train = [self.c]

    def step(self, stim):
        if stim and self.t - self.last_stim > self.tau_ref:
            self.c += self.dt / self.tau_inc
            self.last_stim = self.t
        elif self.t - self.last_stim <= self.tau_stim:
            self.c += self.dt / self.tau_inc

        self.c -= self.dt * (self.c - self.c0) / self.tau_dec
        self.c_train.append(self.c)
        self.t += self.dt

if __name__ == '__main__':
    model = ReducedFastCell()
    for t in np.arange(0, 2, 0.001):
        if t == 0.0:
            model.step(1)
        else:
            model.step(0)

    plt.figure()
    plt.plot(np.arange(0, 2.001, 0.001), model.c_train)
    plt.show()