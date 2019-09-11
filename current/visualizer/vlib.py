#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as anim
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

def save_anim(x, interval, filename, canvas = 'flat', show = True):
    # Convert the 2D data into a video and save it
    fig = plt.figure()
    ims = []

    values = x.values
    time_len = len(values)
    space_len = int(np.sqrt(len(values[0])))

    x = np.reshape(values, (time_len, space_len, space_len))

    if canvas == "flat":
        for j in tqdm(range(np.int(len(x)/interval))):
            im = plt.imshow(x[interval * j], cmap = 'Greens', animated = True, vmin=1, vmax=2.5)
            ims.append([im])

    elif canvas == "cylindar":
        pass

    ani = anim.ArtistAnimation(fig, ims, interval = 1, blit=False, repeat = True)
    ani.save(filename, writer="ffmpeg")
    if show: plt.show()

def save_pattern(x, filename, show = True):
    # Save the 2D/1D data as a spatiotemporal pattern
    ax = plt.figure().gca()
    x = x.T
    im = plt.imshow(x, aspect = 'auto', cmap='Greens')
    plt.ylabel('Cell #')
    plt.xlabel('t [ms]')
    # fig.savefig(filename)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cb = plt.colorbar()
    cb.set_label("Fluorescence [A.U.]")
    if show:    plt.show()

def save_curve(x, filename, show = True):
    # Save the data as curve figure
    fig = plt.figure()
    x = x.values.T
    plt.plot(x[10])
    plt.xlabel('t [ms]')
    plt.ylabel('Fluorescence [A.U.]')
    plt.title('Fluorescence of Cell 10')
    fig.savefig(filename)
    if show:    plt.show()


if __name__ == '__main__':
    
    # x = pd.read_csv('../save/data/c_20x20_100s.csv')
    # # save_pattern(x, '../save/figures/grid_pattern.png')
    # save_anim(x, 1, '../save/animations/grid_movie.mp4')

    x = pd.read_csv('../save/data/fluo_20x1_200s.csv')
    # save_pattern(x, '../save/figures/chain_fluo_pattern.png')
    # save_anim(x, 1, '../save/animations/grid_fluo_movie.mp4')
    save_pattern(x, None)
