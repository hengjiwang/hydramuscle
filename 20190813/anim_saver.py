import matplotlib.pyplot as plt
from tqdm import tnrange
import matplotlib.animation as anim
import numpy as np
import pandas as pd

def save_anim(x, interval, filename, canvas = 'flat'):

    fig = plt.figure()
    ims = []

    if canvas == "flat":
        for j in tnrange(np.int(len(x)/interval)):
            im = plt.imshow(x[interval * j], cmap = 'Greens', animated = True, vmin=1, vmax=2.5)
            ims.append([im])

    elif canvas == "cylindar":
        pass

    ani = anim.ArtistAnimation(fig, ims, interval = 1, blit=False, repeat = True)
    ani.save(filename, writer="ffmpeg")
    plt.show()

if __name__ == "__main__":
    x = pd.read_csv('save/data/fluorescence_2d_diff_endo_freq.csv')
    x = np.reshape(x.values, (20000, 10, 10))
    save_anim(x, 10, 'save/animations/fluorescence_2d_diff_endo_freq.mp4')
