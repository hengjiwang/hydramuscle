import numpy as np

def sig(v, vstar, sstar):
    "Sigmoidal function"
    return 1 / (1 + np.exp(-(v-vstar)/sstar))

def bell(v, vstar, sstar, taustar, tau0):
    "Bell-shape function"
    return taustar/(np.exp(-(v-vstar)/sstar) + np.exp((v-vstar)/sstar)) + tau0

def set_attr(obj, attr, val):
    "A wrapper of setattr(), raises error if attr not exists"
    if not hasattr(obj, attr):
        if not hasattr(obj, '_'+attr):
            raise AttributeError(obj.__class__.__name__ +
                                " has no attribute " +
                                attr)
        else:
            setattr(obj, '_'+attr, val)
    else:
        setattr(obj, attr, val)

def generate_indices(numy, xmin, xmax, ymin, ymax):
    "Generate indices from coordinates range"
    res = []

    for j in range(ymin, ymax):
        for i in range(xmin, xmax):
            res.append(i * numy + j)

    return res
    