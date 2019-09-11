import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from d1.chain import Chain
from fluorescence.fluo_encoder_1d import FluoEncoder1D
import visualizer.vlib

if __name__ == "__main__":

    # Obtain calcium data
    n_cel = 20

    model = Chain(n_cel, 200)
    sol = model.step()
    c = sol[:,0:n_cel]
    s = sol[:,n_cel:2*n_cel]
    r = sol[:,2*n_cel:3*n_cel]
    ip = sol[:,3*n_cel:4*n_cel]
    v = sol[:, 4*n_cel:5*n_cel]

    fig = plt.figure()
    plt.subplot(221)
    model.plot(c, ylabel = 'c[uM]')
    plt.subplot(222)
    model.plot(s, ylabel = 'c_ER[uM]')
    plt.subplot(223)
    model.plot(v, ylabel = 'v[mV]')
    plt.subplot(224)
    model.plot(ip, ylabel = 'IP3[uM]')
    fig.savefig('./save/figures/curves_c_20x1_200s_vstim001.png')
    # plt.show()

    df = pd.DataFrame(c)
    df.to_csv('./save/data/c_20x1_200s_vstim001.csv', index = False)

    # Encode calcium data into fluorescence data
    encoder = FluoEncoder1D(c)
    fluo = encoder.step()
    df = pd.DataFrame(fluo)
    df.to_csv('./save/data/fluo_20x1_200s_vstim001.csv', index = False)

    # Visualize calcium and fluorescence data
    visualizer.vlib.save_pattern(c, './save/figures/chain_c_20x1_200s_vstim001.png', show = False)
    visualizer.vlib.save_pattern(fluo, './save/figures/chain_fluo_20x1_200s_vstim001.png', show = False)