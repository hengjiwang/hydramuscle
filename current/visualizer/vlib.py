#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from tqdm import tnrange
import matplotlib.animation as anim
import numpy as np
import pandas as pd

def save_anim(x, interval, filename, canvas = 'flat', show = True):
    # Convert the 2D data into a video and save it
    fig = plt.figure()
    ims = []

    values = x.values
    time_len = len(values)
    space_len = int(np.sqrt(len(values[0])))

    x = np.reshape(values, (time_len, space_len, space_len))

    if canvas == "flat":
        for j in tnrange(np.int(len(x)/interval)):
            im = plt.imshow(x[interval * j], cmap = 'Greens', animated = True, vmin=1, vmax=2.5)
            ims.append([im])

    elif canvas == "cylindar":
        pass

    ani = anim.ArtistAnimation(fig, ims, interval = 1, blit=False, repeat = True)
    ani.save(filename, writer="ffmpeg")
    if show: plt.show()

def save_pattern(x, filename, show = True):
    # Save the 2D data as a spatiotemporal pattern
    fig = plt.figure()
    x = x.T
    im = plt.imshow(x, aspect = 'auto')
    plt.ylabel('Cell #')
    plt.xlabel('t [ms]')
    fig.savefig(filename)
    if show:    plt.show()

if __name__ == '__main__':
    
    x = pd.read_csv('../save/data/c_50x1_100s.csv')
    save_pattern(x, '../save/figures/chain_pattern.png')
