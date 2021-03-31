import numpy as np
import matplotlib.pyplot as plt

class ReducedFastCell:

    def __init__(self, dt=0.001, neighbors=[]):
        self.dt = dt
        self.t = 0
        self.c0 = 0.05 # uM
        self.tau_stim = 0.05 # s
        self.r_stim = 1 # uM/s
        self.tau_inc = 0.075 # uM/s
        self.tau_dec = 0.5 # s
        self.tau_ref = 0.1 # s
        self.tau_del = 0.01 # s
        self.last_stim = -100 # s
        self.c = self.c0
        self.c_train = [self.c]
        self.v = 0
        self.v_train = [self.v]
        self.neighbors = neighbors

    def step(self, stim=False):

        # Check whether the cell is directly stimulated or stimulated by neighbors
        condition = stim
        if not condition and self.t >= self.tau_del:
            for neighbor in self.neighbors:
                if neighbor.v_train[int((self.t - self.tau_del) / self.dt)]:
                    condition = True
                    break

        # Check if the calcium elevation should be triggered
        if condition and self.t - self.last_stim > self.tau_ref:
            self.v = 1
            self.c += self.dt / self.tau_inc
            self.last_stim = self.t
        elif self.t - self.last_stim <= self.tau_stim:
            self.v = 0
            self.c += self.dt / self.tau_inc
        else:
            self.v = 0

        # Update attributes
        self.c -= self.dt * (self.c - self.c0) / self.tau_dec
        self.c_train.append(self.c)
        self.v_train.append(self.v)
        self.t += self.dt

if __name__ == '__main__':
    model = ReducedFastCell()
    for t in np.arange(0, 2, 0.001):
        if t == 0.0 or t == 1.0:
            model.step(1)
        else:
            model.step(0)

    plt.figure()
    plt.plot(np.arange(0, 2.001, 0.001), model.c_train)
    plt.show()