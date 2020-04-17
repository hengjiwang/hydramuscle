import numpy as np
import matplotlib.pyplot as plt
from hydramuscle.model.fluo_encoder import FluoEncoder

def plot_single_spike(model, sol, tmin1, tmax1, tmin2, tmax2, full_cell=False, fontsize=30, textsize=50, save_fig=True, save_path="../results/figures/fast-pathway.png"):
    "Plot fast pathway"
    index_min = int(tmin1/model.dt)
    index_max = int(tmax1/model.dt)

    if not full_cell:
        c = sol[:, 0]
        v = sol[:, 1]
        m = sol[:, 2]
        h = sol[:, 3]
        n = sol[:, 4]
    else:
        c = sol[:, 0]
        v = sol[:, 4]
        m = sol[:, 5]
        h = sol[:, 6]
        n = sol[:, 7]

    plt.figure(figsize=(30,10), tight_layout=True)

    ax1 = plt.subplot2grid((1,3), (0,1), colspan=1)
    ax1.plot(model.time[index_min:index_max]*1000, c[index_min:index_max], linewidth=5, color="k")
    ax1.tick_params(labelsize=fontsize)
    ax1.set_xlabel("time(ms)", fontsize=fontsize)
    ax1.set_ylabel(r"[Ca$^{2+}$](uM)", fontsize=fontsize)
    ax1.text(-0.01, 1.05, 'B', size=textsize, weight="bold", transform=ax1.transAxes)

    ax2 = plt.subplot2grid((1,3), (0,0), colspan=1)
    ax2.plot(model.time[index_min:index_max]*1000, v[index_min:index_max], linewidth=5, color="k")
    ax2.tick_params(labelsize=fontsize)
    ax2.set_xlabel("time(ms)", fontsize=fontsize)
    ax2.set_ylabel("Membrane voltage(mV)", fontsize=fontsize)
    ax2.text(-0.01, 1.05, 'A', size=textsize, weight="bold", transform=ax2.transAxes)

    index_min = int(tmin2/model.dt)
    index_max = int(tmax2/model.dt)

    ax3 = plt.subplot2grid((1,3), (0,2), colspan=1)
    ax3.plot(model.time[index_min:index_max]*1000, model.i_ca(v, m, h)[index_min:index_max], linewidth=5, color="r", label=r"I$_{Ca}$")
    ax3.plot(model.time[index_min:index_max]*1000, model.i_k(v, n)[index_min:index_max], linewidth=5, color="b", label=r"I$_{K}$")
    ax3.plot(model.time[index_min:index_max]*1000, model.i_bk(v)[index_min:index_max], linewidth=5, color="purple", linestyle="--", label=r"I$_{L}$")
    ax3.legend(fontsize=fontsize, loc='upper right')
    ax3.tick_params(labelsize=fontsize)
    ax3.set_xlabel("time(ms)", fontsize=fontsize)
    ax3.set_ylabel(r"Membrane current(mA/cm$^2$)", fontsize=fontsize)
    ax3.text(-0.005, 1.05, 'C', size=textsize, weight="bold", transform=ax3.transAxes)
    ax3.bar(index_min, 0.0005, width=10, bottom=-0.015, align='edge', color='k')

    if save_fig:   
        plt.savefig(save_path)

    plt.show()

def plot_slow_transient(model, sol, tmin, tmax, full_cell=False, fontsize=30, textsize=50, save_fig=True, save_path="../results/figures/slow-pathway.png"):
    "Plot slow pathway"
    index_min = int(tmin/model.dt)
    index_max = int(tmax/model.dt)

    c = sol[:,0]
    s = sol[:,1]
    r = sol[:,2]
    ip = sol[:,3]

    plt.figure(figsize=(25,10), tight_layout=True)

    ax1 = plt.subplot2grid((1,2), (0,0), colspan=1)
    ax1.plot(model.time[index_min:index_max], c[index_min:index_max], linewidth=5, color="k")
    ax1.tick_params(axis='y', labelsize=fontsize, labelcolor='k')
    ax1.set_xlabel("time(s)", fontsize=fontsize)
    ax1.set_ylabel(r"[Ca$^{2+}$](uM)", fontsize=fontsize)
    ax1.text(-0.01, 1.05, 'A', size=textsize, weight="bold", transform=ax1.transAxes)

    ax3 = ax1.twinx()
    ax3.plot(model.time[index_min:index_max], ip[index_min:index_max], linewidth=5, color="r", linestyle="--")
    ax3.tick_params(axis='y', labelsize=fontsize, labelcolor='r')
    ax3.set_ylim(0,10)
    ax3.set_ylabel(r"[IP$_3$]", fontsize=fontsize, color='r')
    # ax3.text(-0.01, 1.05, 'B', size=40, weight="bold", transform=ax3.transAxes)

    ax2 = plt.subplot2grid((1,2), (0,1), colspan=1)
    # ax2.plot(model.time[index_min:index_max], ip[index_min:index_max], linewidth=5, color="k", linestyle="--", label=r"IP$_3$")
    ax2.plot(model.time[index_min:index_max], model.i_ipr(c, s, ip, r)[index_min:index_max], linewidth=5, color="r", label=r"J$_{IPR}$")
    ax2.plot(model.time[index_min:index_max], -model.i_serca(c)[index_min:index_max], linewidth=5, color="g", label=r"J$_{SERCA}$")
    ax2.plot(model.time[index_min:index_max], -model.i_pmca(c)[index_min:index_max], linewidth=5, color="b", label=r"J$_{PMCA}$")
    ax2.plot(model.time[index_min:index_max], model.i_leak(c, s)[index_min:index_max], linewidth=5, color="y", label=r"J$_{leak}$")
    if not full_cell:
        ax2.plot(model.time[index_min:index_max], model.i_in(ip)[index_min:index_max], linewidth=5, color="cyan", label=r"J$_{in}$")
    else:
        v = sol[:,4]
        m = sol[:,5]
        h = sol[:,6]
        ax2.plot(model.time[index_min:index_max], -model.alpha*model.i_ca(v, m, h)[index_min:index_max] + model.i_in(ip)[index_min:index_max], linewidth=5, color="purple", label=r"-$\alpha$I$_{Ca}$+J$_{in}$")
    ax2.tick_params(labelsize=fontsize)
    ax2.set_xlabel("time(s)", fontsize=fontsize)
    ax2.set_ylabel(r"Ca$^{2+}$ Fluxes ($\mu$M/s)", fontsize=fontsize)
    ax2.text(-0.01, 1.05, 'B', size=textsize, weight="bold", transform=ax2.transAxes)
    ax2.legend(fontsize=fontsize, loc='upper right')

    # ax3 = plt.subplot2grid((1,3), (0,1), colspan=1)
    # ax3.plot(model.time[index_min:index_max], ip[index_min:index_max], linewidth=5, color="k", linestyle="--")
    # ax3.tick_params(labelsize=20)
    # ax3.set_xlabel("time(ms)", fontsize=20)
    # ax3.set_ylabel(r"[IP$_3$]", fontsize=20)
    # ax3.text(-0.01, 1.05, 'B', size=40, weight="bold", transform=ax3.transAxes)

    if save_fig:   
        plt.savefig(save_path)

    plt.show()

def plot_multiple_spikes(model, sol, force_ecto, force_endo, tmin1, tmax1, tmin2, tmax2, fontsize=30, textsize=50, save_fig=True, save_path="../results/figures/multiple-fast.png"):
    "Plot multiple spikes"
    index_min = int(tmin1/model.dt)
    index_max = int(tmax1/model.dt)


    c = sol[:, 0]
    v = sol[:, 4]

    # Encode calcium into fluorescence
    fluo_encoder = FluoEncoder(c, T=model.T, dt=model.dt)
    fluo = fluo_encoder.step()
    fluo = (fluo - min(fluo))/(max(fluo) - min(fluo))

    plt.figure(figsize=(30,10), tight_layout=True)

    # Plot [Ca2+] and fluoresence in one subplot
    ax1 = plt.subplot2grid((1,3), (0,1), colspan=1)
    ax1.plot(model.time[index_min:index_max], c[index_min:index_max], linewidth=5, color="k")
    ax1.tick_params(labelsize=fontsize)
    ax1.set_xlabel("time(s)", fontsize=fontsize)
    ax1.set_ylabel(r"[Ca$^{2+}$](uM)", fontsize=fontsize)
    ax1.text(-0.01, 1.05, 'B', size=textsize, weight="bold", transform=ax1.transAxes)

    ax3 = ax1.twinx()
    ax3.plot(model.time[index_min:index_max], fluo[index_min:index_max], linewidth=5, color="purple")
    ax3.tick_params(axis='y', labelsize=fontsize, labelcolor='purple')
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Fluorescence(a.u.)", fontsize=fontsize, color='purple')

    # Plot membrane potential 
    ax2 = plt.subplot2grid((1,3), (0,0), colspan=1)
    ax2.plot(model.time[index_min:index_max], v[index_min:index_max], linewidth=5, color="k")
    ax2.tick_params(labelsize=fontsize)
    ax2.set_xlabel("time(s)", fontsize=fontsize)
    ax2.set_ylabel("Membrane voltage(mV)", fontsize=fontsize)
    ax2.text(-0.01, 1.05, 'A', size=textsize, weight="bold", transform=ax2.transAxes)

    # Plot active force
    index_min = int(tmin2/model.dt)
    index_max = int(tmax2/model.dt)

    ax3 = plt.subplot2grid((1,3), (0,2), colspan=1)
    ax3.plot(model.time[index_min:index_max], force_ecto[index_min:index_max], linewidth=5, color="g", label=r"Ectoderm")
    ax3.plot(model.time[index_min:index_max], force_endo[index_min:index_max], linewidth=5, color="r", label=r"Endoderm")
    ax3.legend(fontsize=fontsize, loc='upper right')
    ax3.tick_params(labelsize=fontsize)
    ax3.set_xlabel("time(s)", fontsize=fontsize)
    ax3.set_ylabel("Active force(a.u.)", fontsize=fontsize)
    ax3.text(-0.005, 1.05, 'C', size=textsize, weight="bold", transform=ax3.transAxes)
    # ax3.bar(index_min, 0.0005, width=5, bottom=-0.015, align='edge', color='k')

    if save_fig:   
        plt.savefig(save_path)

    plt.show()
