import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt

try:
    from cv2 import cv2
except:
    import cv2
from tqdm import tqdm
import scipy.ndimage


def euler_odeint(rhs, y, T, dt, save_interval=1, **kwargs):
    "An Euler-method integrator"

    sol = np.zeros((int(T/dt/save_interval), len(y)))

    for j in tqdm(np.arange(0, int(T/dt))):
        if j % save_interval == 0:
            sol[int(j/save_interval), :] = y
        t = j*dt
        dydt = rhs(y, t, **kwargs)
        y += dydt * dt

    return sol


def euler_odeint2(rhs, y, T, dt, save_interval=1, numx=200, numy=200, layer_num=2, numvar=8, **kwargs):
    "An Euler-method integrator"

    sol = np.zeros((int(T/dt/save_interval), layer_num*numx*numy))

    for j in tqdm(np.arange(0, int(T/dt))):
        if j % save_interval == 0:
            sol[int(j/save_interval), 0:numx*numy] = y[0:numx*numy]
            if layer_num == 2:
                sol[int(j/save_interval), numx*numy:2*numx*numy] = y[numvar*numx*numy:(numvar+1)*numx*numy]
        t = j*dt
        dydt = rhs(y, t, **kwargs)
        y += dydt * dt

    return sol


def sig(v, vstar, sstar):
    "Sigmoidal function"
    return 1 / (1 + np.exp(-(v - vstar) / sstar))


def bell(v, vstar, sstar, taustar, tau0):
    "Bell-shape function"
    return taustar / (np.exp(-(v - vstar) / sstar) + np.exp((v - vstar) / sstar)) + tau0


def set_attr(obj, attr, val):
    "A wrapper of setattr(), raises error if attr not exists"
    if not hasattr(obj, attr):
        if not hasattr(obj, '_' + attr):
            raise AttributeError(obj.__class__.__name__ +
                                 " has no attribute " +
                                 attr)
        setattr(obj, '_' + attr, val)
    else:
        setattr(obj, attr, val)


def generate_indices(numy, xmin, xmax, ymin, ymax):
    "Generate indices from coordinates range"
    res = []

    for j in range(ymin, ymax):
        for i in range(xmin, xmax):
            res.append(i * numy + j)

    return res


def generate_random_indices(numx, numy, randomnum, neighborsize):
    "Generate <=9*randomnum indices in a numx*numy matrix"
    res = []
    for _ in range(randomnum):

        # Middle point
        # np.random.seed(1)
        indx = np.random.randint(0, numx)
        # np.random.seed(2)
        indy = np.random.randint(0, numy)

        # Neighboring points
        for dx in range(-neighborsize // 2 + 1, neighborsize // 2 + 1):
            for dy in range(-neighborsize // 2 + 1, neighborsize // 2 + 1):
                indx_ = indx + dx
                indy_ = indy + dy

                if 0 <= indx_ < numx and 0 <= indy_ < numy:
                    res.append(indx_ * numy + indy_)

    return list(set(res))


def track_wavefront(data, thres, pathway='slow'):
    "Track the wavefront of trace data"

    ntime = len(data)
    numcell = len(data[0])

    wavefront = np.zeros(ntime)

    if pathway == 'slow':

        for j in range(ntime):

            if j < int(20 / 0.02):
                wavefront[j] = 0
                continue

            for k in range(numcell - 1, -1, -1):
                if 2 <= k < 30 and data[j][k] > thres and data[j][k] - data[j][k + 20] > 0.1:
                    wavefront[j] = k if (0 < k - wavefront[j - 1] < 4 or wavefront[j - 1] == 0) else wavefront[j - 1]
                    break

    elif pathway == 'fast':

        for j in range(ntime):
            for k in range(numcell - 1, -1, -1):
                if data[j][k] > thres:
                    wavefront[j] = k
                    break

    return wavefront


def compress_frame(frame, size_x, size_y, to_size_x, to_size_y):
    "Compress the frame by averaging neighboring elements"
    mat = np.reshape(frame, (size_x, size_y))
    new_mat = np.zeros((to_size_x, to_size_y))
    win_x = int(size_x / to_size_x)
    win_y = int(size_y / to_size_y)

    for i in range(to_size_x):
        for j in range(to_size_y):
            new_mat[i, j] = np.mean(mat[i * win_x:(i + 1) * win_x, j * win_y:(j + 1) * win_y])

    return np.reshape(new_mat, (to_size_x * to_size_y))


def average_force(force, numx, numy, to_numx, to_numy):
    "Average the force to 20x20"

    force_averaged = np.zeros((len(force), to_numx * to_numy))

    for j in range(len(force)):
        frame = force[j]

        force_averaged[j, :] = compress_frame(frame, numx, numy, to_numx, to_numy)

    return np.reshape(force_averaged, (-1, to_numx, to_numy))


def encode_force_2d(encoder, c, numx, numy, dt, save_interval=1):
    "Encode calcium concentration matrix c into force"

    c = c.reshape(-1, numx * numy)

    def _rhs(y, t, c, num2):
        "Right-hand side"

        j = int(t / dt)

        k1 = c[j] ** encoder.nm / (c[j] ** encoder.nm + encoder.c_half ** encoder.nm)
        encoder.k6 = k1

        m, mp, amp, am = y[0:num2], y[num2:2 * num2], y[2 * num2:3 * num2], y[3 * num2:4 * num2]

        dmdt = -k1 * m + encoder.k2 * mp + encoder.k7 * am
        dmpdt = k1 * m + (-encoder.k2 - encoder.k3) * mp + encoder.k4 * amp
        dampdt = encoder.k3 * mp + (-encoder.k4 - encoder.k5) * amp + encoder.k6 * am
        damdt = encoder.k5 * amp + (-encoder.k6 - encoder.k7) * am

        dydt = np.reshape([dmdt, dmpdt, dampdt, damdt], 4 * num2)

        return dydt

    T = len(c) * dt

    num2 = numx * numy
    base_mat = np.ones((numy, numx))
    inits = [encoder.m0, encoder.mp0, encoder.amp0, encoder.am0]
    y0 = np.array([x * base_mat for x in inits])
    y0 = np.reshape(y0, 4 * num2)

    sol = euler_odeint(rhs=_rhs,
                       y=y0,
                       T=T,
                       dt=dt,
                       save_interval=save_interval,
                       c=c,
                       num2=num2)

    force = encoder.K * (sol[:, 2 * num2:3 * num2] + sol[:, 3 * num2:4 * num2])

    return force.reshape(-1, numx, numy)


def save_video(filename, savepath, numx=200, numy=200, flip=True, dpi=50, fps=200):
    "Convert data to video"
    data = pd.read_hdf(filename)
    data = data.values.reshape(-1, numx, numy)

    plt.figure(figsize=(numx / dpi, numy / dpi))

    for iframe in tqdm(range(len(data))):
        plt.clf()

        frame = data[iframe]
        # if flip:
        #     frame = np.flip(frame.T, 0)

        plt.imshow(frame.T, cmap='hot', vmin=0, vmax=2)

        plt.axis('off')

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.xlim(0, numx)
        plt.ylim(0, numy)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.savefig(savepath + 'frames/img' + str(iframe) + '.jpg', dpi=dpi)

    # Save video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter(savepath + '/video.avi', fourcc, fps, (numx, numy))

    for iframe in tqdm(range(len(data))):
        frame = cv2.imread(savepath + 'frames/img' + str(iframe) + '.jpg')
        videoWriter.write(frame)
    videoWriter.release()
    cv2.destroyAllWindows()


def length_of_model(coordspath, totaltime=300, loc='x', display=True, ret_midpts=False):
    "Extract the length trace of model from the coordinates of sidepoints"
    # Get number of points and time steps
    count = 0
    ntime = 0
    with open(coordspath, 'r') as fp:
        while True:
            count += 1
            line = fp.readline()
            if not line:
                break
            if count == 7:
                npoints = len(line.split())
            if count >= 6:
                ntime += 1

    ntime = ntime // 3

    # Reformat coordinates
    mat = np.zeros((ntime, npoints, 3))

    count = 0
    itime = -1
    with open(coordspath, 'r') as fp:
        while True:
            count += 1
            line = fp.readline()
            if not line:
                break

            if count < 6:
                pass
            elif count % 3 == 0:
                itime += 1
                coords = line.split()[1:]
                coords = [float(x) for x in coords]
                mat[itime, :, 0] = coords
            elif count % 3 == 1:
                coords = line.split()
                coords = [float(x) for x in coords]
                mat[itime, :, 1] = coords
            else:
                coords = line.split()
                coords = [float(x) for x in coords]
                mat[itime, :, 2] = coords

    # Change the unit from m to mm
    mat *= 1000

    # Divide positive and negative points
    mat_pos = np.zeros((ntime, npoints // 2, 3))
    mat_neg = np.zeros((ntime, npoints // 2, 3))

    ipos = 0
    ineg = 0

    division = 1 if loc == 'x' else 0 if loc == 'y' else None

    for j in range(len(mat[0])):
        if mat[division][j][0] < 0:
            mat_neg[:, ineg, :] = mat[:, j, :]
            ineg += 1
        else:
            mat_pos[:, ipos, :] = mat[:, j, :]
            ipos += 1

    # Sort the points from bottom to top

    #     z_original = mat_pos[0, :, 2]
    z_original = mat_pos[0, :, 2] ** 2 + mat_pos[0, :, 1] ** 2 + mat_pos[0, :, 0] ** 2
    argsort = np.argsort(z_original)
    mat_pos_sorted = mat_pos[:, argsort, :]

    #     z_original = mat_neg[0, :, 2]
    z_original = mat_neg[0, :, 2] ** 2 + mat_neg[0, :, 1] ** 2 + mat_neg[0, :, 0] ** 2
    argsort = np.argsort(z_original)
    mat_neg_sorted = mat_neg[:, argsort, :]

    # Get the middle points
    mat_mid = (mat_pos_sorted + mat_neg_sorted) / 2

    # Calculate the length
    lengths = []

    for j in range(len(mat_mid)):
        points = mat_mid[j]
        length = 0
        for k in range(len(points) - 1):
            diffvec = points[k + 1] - points[k]
            dist = LA.norm(diffvec)
            length += dist
        lengths.append(length)

    if display:
        plt.figure(figsize=(int(totaltime / 30), 3))
        plt.plot(np.arange(0, totaltime + 0.1, 0.1), lengths, color='b', linewidth=2)
        plt.xlabel('time(s)')
        plt.ylabel('length(mm)')
        plt.show()

    if ret_midpts:
        return lengths, mat_mid
    else:
        return lengths


def normalize(seq):
    "Normalize a sequence"
    minval = min(seq)
    maxval = max(seq)
    return [(x - minval) / (maxval - minval) for x in seq]


def filter_abnormal(seq, size=50, thres=0.2):
    "Filter out abnormal points in seq"

    res = []
    seq_smooth = scipy.ndimage.filters.median_filter(seq, size=size)

    maxval = max(seq)
    minval = min(seq)

    for j in range(len(seq)):
        val = seq[j]
        val_smooth = seq_smooth[j]
        if abs(val - val_smooth) > thres * (maxval - minval):
            res.append(val_smooth)
        else:
            res.append(val)

    return res