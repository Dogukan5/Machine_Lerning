import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    N = len(samples)
    #  estDensity : estimated density in the range of [-5,5]
    pos = np.arange(-5,5.0,0.1)
    insE = -((pos[np.newaxis, :] - samples[:, np.newaxis]) ** 2)/(2* pow(h,2))
    norm = 1 / (pow(2 * np.pi*pow(h,2),0.5) *N)
    res = np.sum(norm * np.exp(insE), axis=0)
    estDensity = np.stack((pos, res), axis=1)
    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created


    # Form the output variable
    estDensity = np.stack((pos, res), axis=1)
    return estDensity
