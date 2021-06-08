import numpy as np


def get_kf(boxsize):
    """Returns the Fundamental Fourier mode.

    Parameters
    ----------
    boxsize : float
        Box size.

    Returns
    -------
    kf : float
        Fundamental mode.
    """
    kf = 2.*np.pi/boxsize
    return kf


def get_kn(boxsize, ngrid):
    """Returns the Nyquist Fourier mode.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid size.

    Returns
    -------
    kn : float
        Nyquist frequency.
    """
    kn = ngrid*np.pi/boxsize
    return kn
