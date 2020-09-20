import numpy as np
import matplotlib.pyplot as plt
from hydra.model.fluo_encoder import FluoEncoder
import hydra.model.helper as helper
from matplotlib.ticker import FuncFormatter

def plot_single_spike(model, sol, tmin1, tmax1, tmin2, tmax2, full_cell=False, 
                      fontsize=30, textsize=50, save_fig=True, 
                      save_path="../results/figures/fast-pathway.png"):
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

    plt.figure(figsize=(18, 5), tight_layout=True)

    ax1 = plt.subplot2grid((1, 3), (0, 1), colspan=1)
    ax1.plot(model.time[index_min:index_max]*1000, c[index_min:index_max], linewidth=5, color="k")
    ax1.tick_params(labelsize=fontsize)
    ax1.set_xlabel("time(ms)", fontsize=fontsize)
    ax1.set_ylabel(r"[Ca$^{2+}$](uM)", fontsize=fontsize)
    ax1.text(-0.01, 1.05, 'B', size=textsize, weight="bold", transform=ax1.transAxes)

    ax2 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
    ax2.plot(model.time[index_min:index_max]*1000, v[index_min:index_max], linewidth=5, color="k")
    ax2.tick_params(labelsize=fontsize)
    ax2.set_xlabel("time(ms)", fontsize=fontsize)
    ax2.set_ylabel("Membrane voltage(mV)", fontsize=fontsize)
    ax2.text(-0.01, 1.05, 'A', size=textsize, weight="bold", transform=ax2.transAxes)

    index_min = int(tmin2/model.dt)
    index_max = int(tmax2/model.dt)

    ax3 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
    ax3.plot(model.time[index_min:index_max]*1000, model.i_ca(v, m, h)[index_min:index_max],
             linewidth=5, color="r", label=r"I$_{Ca}$")
    ax3.plot(model.time[index_min:index_max]*1000, model.i_k(v, n)[index_min:index_max],
             linewidth=5, color="b", label=r"I$_{K}$")
    ax3.plot(model.time[index_min:index_max]*1000, model.i_bk(v)[index_min:index_max],
             linewidth=5, color="purple", linestyle="--", label=r"I$_{L}$")
    ax3.legend(fontsize=fontsize, loc='upper right')
    ax3.tick_params(labelsize=fontsize)
    ax3.set_xlabel("time(ms)", fontsize=fontsize)
    ax3.set_ylabel(r"Membrane current(mA/cm$^2$)", fontsize=fontsize)
    ax3.text(-0.005, 1.05, 'C', size=textsize, weight="bold", transform=ax3.transAxes)
    ax3.bar(index_min, 0.0005, width=10, bottom=-0.015, align='edge', color='k')

    if save_fig:   
        plt.savefig(save_path)

    plt.show()

def plot_slow_transient(model, sol, tmin, tmax, full_cell=False, fontsize=30, textsize=50,
                        save_fig=True, save_path="../results/figures/slow-pathway.png"):
    "Plot slow pathway"
    index_min = int(tmin/model.dt)
    index_max = int(tmax/model.dt)

    c = sol[:, 0]
    s = sol[:, 1]
    r = sol[:, 2]
    ip = sol[:, 3]

    plt.figure(figsize=(14, 5), tight_layout=True)

    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
    ax1.plot(model.time[index_min:index_max], c[index_min:index_max], linewidth=5, color="k")
    ax1.tick_params(axis='x', labelsize=fontsize, labelcolor='k')
    ax1.tick_params(axis='y', labelsize=fontsize, labelcolor='k')
    ax1.set_xlabel("time(s)", fontsize=fontsize)
    ax1.set_ylabel(r"[Ca$^{2+}$](uM)", fontsize=fontsize)
    ax1.text(-0.01, 1.05, 'A', size=textsize, weight="bold", transform=ax1.transAxes)

    ax3 = ax1.twinx()
    ax3.plot(model.time[index_min:index_max], ip[index_min:index_max],
             linewidth=5, color="r", linestyle="--")
    ax3.tick_params(axis='y', labelsize=fontsize, labelcolor='r')
    ax3.set_ylim(0, 10)
    ax3.set_ylabel(r"[IP$_3$]", fontsize=fontsize, color='r')
    # ax3.text(-0.01, 1.05, 'B', size=40, weight="bold", transform=ax3.transAxes)

    ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1)
    # ax2.plot(model.time[index_min:index_max], ip[index_min:index_max],
    #          linewidth=5, color="k", linestyle="--", label=r"IP$_3$")
    ax2.plot(model.time[index_min:index_max], model.i_ipr(c, s, ip, r)[index_min:index_max],
             linewidth=5, color="r", label=r"J$_{IPR}$")
    ax2.plot(model.time[index_min:index_max], -model.i_serca(c)[index_min:index_max],
             linewidth=5, color="g", label=r"J$_{SERCA}$")
    ax2.plot(model.time[index_min:index_max], -model.i_pmca(c)[index_min:index_max],
             linewidth=5, color="b", label=r"J$_{PMCA}$")
    ax2.plot(model.time[index_min:index_max], model.i_leak(c, s)[index_min:index_max],
             linewidth=5, color="y", label=r"J$_{leak}$")
    if not full_cell:
        ax2.plot(model.time[index_min:index_max], model.i_in(ip)[index_min:index_max],
                 linewidth=5, color="cyan", label=r"J$_{in}$")
    else:
        v = sol[:, 4]
        m = sol[:, 5]
        h = sol[:, 6]
        ax2.plot(model.time[index_min:index_max],
                 -model.alpha*model.i_ca(v, m, h)[index_min:index_max] +
                 model.i_in(ip)[index_min:index_max],
                 linewidth=5, color="purple", label=r"-$\alpha$I$_{Ca}$+J$_{in}$")
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

def plot_multiple_spikes(model, sol, force_ecto, force_endo, tmin1, tmax1, tmin2, tmax2,
                         fontsize=30, textsize=50, save_fig=True,
                         save_path="../results/figures/multiple-fast.png"):
    "Plot multiple spikes"
    index_min = int(tmin1/model.dt)
    index_max = int(tmax1/model.dt)


    c = sol[:, 0]
    v = sol[:, 4]

    # Encode calcium into fluorescence
    fluo_encoder = FluoEncoder(c, T=model.T, dt=model.dt)
    fluo = fluo_encoder.step()
    fluo = (fluo - min(fluo))/(max(fluo) - min(fluo))

    plt.figure(figsize=(20, 5), tight_layout=True)

    # Plot [Ca2+] and fluoresence in one subplot
    ax1 = plt.subplot2grid((1, 3), (0, 1), colspan=1)
    ax1.plot(model.time[index_min:index_max], c[index_min:index_max], linewidth=3, color="k")
    ax1.tick_params(labelsize=fontsize)
    ax1.set_xlabel("time(s)", fontsize=fontsize)
    ax1.set_ylabel(r"[Ca$^{2+}$](uM)", fontsize=fontsize)
    ax1.text(-0.01, 1.05, 'B', size=textsize, weight="bold", transform=ax1.transAxes)

    ax3 = ax1.twinx()
    ax3.plot(model.time[index_min:index_max], fluo[index_min:index_max],
             linewidth=5, color="green")
    ax3.tick_params(axis='y', labelsize=fontsize, labelcolor='green')
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("Fluorescence(a.u.)", fontsize=fontsize, color='green')

    # Plot membrane potential
    ax2 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
    ax2.plot(model.time[index_min:index_max], v[index_min:index_max], linewidth=5, color="k")
    ax2.tick_params(labelsize=fontsize)
    ax2.set_xlabel("time(s)", fontsize=fontsize)
    ax2.set_ylabel("Membrane voltage(mV)", fontsize=fontsize)
    ax2.text(-0.01, 1.05, 'A', size=textsize, weight="bold", transform=ax2.transAxes)

    # Plot active force
    index_min = int(tmin2/model.dt)
    index_max = int(tmax2/model.dt)

    ax3 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
    ax3.plot(model.time[index_min:index_max], force_ecto[index_min:index_max],
             linewidth=4, color="darkgreen", label=r"Ectoderm")
    ax3.plot(model.time[index_min:index_max], force_endo[index_min:index_max],
             linewidth=4, color="r", label=r"Endoderm")
    ax3.legend(fontsize=fontsize, loc='upper right')
    ax3.tick_params(labelsize=fontsize)
    ax3.set_xlabel("time(s)", fontsize=fontsize)
    # def formatnum(x, pos):
    #     return '$%.1f$x$10^{5}$' % (x/100000)
    # formatter = FuncFormatter(formatnum)
    # ax3.yaxis.set_major_formatter(formatter)
    ax3.set_ylabel("Active stress(N/mm$^2$)", fontsize=fontsize)
    ax3.text(-0.005, 1.05, 'C', size=textsize, weight="bold", transform=ax3.transAxes)
    # ax3.bar(index_min, 0.0005, width=5, bottom=-0.015, align='edge', color='k')

    if save_fig:
        plt.savefig(save_path)

    plt.show()

def plot_frame_patterns(data, time_pts, vmin, vmax, dt=1):
    "Plot specified frames of data"

    # Get plot dimensions
    nplots = len(time_pts)
    nrow = 2
    ncol = int(np.ceil(nplots/nrow))

    # Plot
    fig = plt.figure(figsize=(20, 10))

    for j in range(nplots):
        ax = fig.add_subplot(nrow, ncol, j+1)
        timep = time_pts[j]
        frame = np.flip(data[int(timep/dt)].T, 0)
        ax.imshow(frame, cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_title('t=' + str(round(timep, 2)) + 's')

    plt.tight_layout()
    plt.show()

def plot_1d_traces(data, interval, dt):
    "Plot calcium traces in different directions"

    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    # Get the data dimensions
    nframes = len(data)
    numx = len(data[0])
    # numy = len(data[0][0])

    # Get data traces
    datax = data[:, numx//2:numx:interval, 0]
    datay = data[:, numx//2, ::interval]

    # Plot
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot([x*dt for x in range(nframes)], datax)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel(r"[Ca$^{2+}$] ($\mu$M)")
    ax1.set_title("x direction")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot([x*dt for x in range(nframes)], datay)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel(r"[Ca$^{2+}$] ($\mu$M)")
    ax2.set_title("y direction")

def plot_slowwave_stills(data, times, dt,
                         save_fig=True, save_path="../results/figures/simulate-slow-wave.png"):
    "Plot the stills of the slow wave"
    fig = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.1)

    # for j in range(len(times)):
    #     ax = fig.add_subplot(2, len(times), j+1)
    #     im = ax.imshow(np.flip(data[int(times[j]/dt)].T, 0), cmap='hot', vmin=0, vmax=1)
    #     ax.text(2, 10, str(times[j]) + 's', color='white', fontsize=20)
    #     # ax.set_xticks([0, 50, 100, 150, 200])
    #     # ax.set_yticks([0, 50, 100, 150, 200])
    #     ax.tick_params(labelsize=15, labelcolor='k')
    #     ax.patch.set_edgecolor('g')
    #     ax.patch.set_alpha(1)  
    #     ax.patch.set_linewidth('15')

    #     if j == 0:
    #         ax.text(-0.45, 1.05, 'A', size=30, weight="bold", transform=ax.transAxes)

    # cax = fig.add_axes([0.92, 0.565, 0.01, 0.285])
    # cb = plt.colorbar(im, cax=cax)
    # font = {'size':15}
    # cb.ax.tick_params(labelsize=15)
    # cb.set_label(r"[Ca$^{2+}$] ($\mu$M)", fontdict=font)

    ax2 = fig.add_subplot(2, 2, 3)
    time_axis = np.arange(20, 80, dt)
    ax2.plot(time_axis, data[int(20/dt):int(80/dt), 15, ::1], linewidth=1.5)
    ax2.set_xlabel("time (s)", fontsize=15)
    ax2.set_ylabel(r"[Ca$^{2+}$] ($\mu$M)", fontsize=15)
    ax2.tick_params(labelsize=15, labelcolor='k')
    # ax2.text(-0.1, 1.0, 'B', size=30, weight="bold", transform=ax2.transAxes)

    ax3 = fig.add_subplot(2, 2, 4)

    center_col = data[:, 15, :]
    wavefront = helper.track_wavefront(center_col, 0.1)

    # print(list(wavefront))

    startp = np.where(wavefront == 6)[0][0]*dt
    endp = np.where(wavefront == max(wavefront))[0][0]*dt

    wavefront = wavefront[int(20/dt):int(80/dt)]

    ax3.axvline(startp, linestyle='--', color='r')
    ax3.axvline(endp, linestyle='--', color='r')

    ax3.text(startp, 0, str(startp), size=15, color='r')
    ax3.text(endp, 0, str(endp), size=15, color='r')

    time_axis = np.arange(20, 80, dt)
    ax3.plot(time_axis, wavefront, 'b', linewidth=3)
    ax3.grid()
    ax3.set_xlabel("time (s)", fontsize=15)
    ax3.set_ylabel("Wave front #", fontsize=15)
    ax3.tick_params(labelsize=15, labelcolor='k')

    # ax3.axhline(max(wavefront), linestyle='--', color='g')
    # ax3.axvline(startp*dt, linestyle='--', color='r')
    # ax3.axvline(endp*dt, linestyle='--', color='r')
    # ax3.text(startp*dt, -3, str(round(startp*dt,1)), size=15, color='r')
    # ax3.text(endp*dt, -3, str(round(endp*dt,1)), size=15, color='r')


    # ax3.text(-0.1, 1.0, 'C', size=30, weight="bold", transform=ax3.transAxes)
    # ax3.text(-6.3, max(wavefront)-2, str(int(max(wavefront))), size=15, color='g')
    

    # plt.tight_layout()
    if save_fig:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_fastwave_stills(data, times, dt, endtime,
                         save_fig=True, save_path="../results/figures/simulate-fast-wave.png"):
    "Plot the stills of the slow wave"
    fig = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.1)

    # for j in range(len(times)):
    #     ax = fig.add_subplot(2, len(times), j+1)
    #     im = ax.imshow(np.flip(data[int(times[j]/dt)].T, 0), cmap='hot', vmin=0, vmax=1)
    #     ax.text(2, 5, str(int(times[j]*1000)) + 'ms', color='white', fontsize=20)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.tick_params(labelsize=15, labelcolor='k')
    #     # if j == 0:
    #     #     ax.text(-0.45, 1.05, 'D', size=30, weight="bold", transform=ax.transAxes)
    #     ax.patch.set_edgecolor('g')  
    #     ax.patch.set_alpha(1)
    #     ax.patch.set_linewidth('15')
    

    # cax = fig.add_axes([0.92, 0.55, 0.01, 0.33])
    # cb = plt.colorbar(im, cax=cax)
    # font = {'size':15}
    # cb.ax.tick_params(labelsize=15)
    # cb.set_label(r"[Ca$^{2+}$] ($\mu$M)", fontdict=font)

    ax2 = fig.add_subplot(2, 2, 3)
    time_axis = np.arange(0, endtime*1000, dt*1000)
    ax2.plot(time_axis, data[:int(endtime/dt), 25, ::5], linewidth=3)
    ax2.set_xlabel("time (ms)", fontsize=15)
    ax2.set_ylabel(r"[Ca$^{2+}$] ($\mu$M)", fontsize=15)
    ax2.tick_params(labelsize=15, labelcolor='k')
    # ax2.text(-0.1, 1.05, 'E', size=30, weight="bold", transform=ax2.transAxes)

    ax3 = fig.add_subplot(2, 2, 4)

    center_col = data[:int(endtime/dt), 25, :]
    wavefront = helper.track_wavefront(center_col, 0.1, 'fast')

    # print(len(wavefront))

    startp = np.where(wavefront == 1)[0][0]*(dt*1000)
    endp = np.where(wavefront == 59)[0][0]*(dt*1000)

    ax3.axvline(startp, linestyle='--', color='r')
    ax3.axvline(endp, linestyle='--', color='r')

    ax3.text(startp, -5, str(round(startp,2)), size=15, color='r')
    ax3.text(endp, -5, str(round(endp,2)), size=15, color='r')

    ax3.plot(time_axis, wavefront, 'b', linewidth=3)
    ax3.grid()
    ax3.set_xlabel("time (ms)", fontsize=15)
    ax3.set_ylabel("Wave front #", fontsize=15)
    ax3.tick_params(labelsize=15, labelcolor='k')
    # ax3.text(-0.1, 1.05, 'F', size=30, weight="bold", transform=ax3.transAxes)

    # plt.tight_layout()
    if save_fig:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()