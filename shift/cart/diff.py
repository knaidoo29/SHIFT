import numpy as np


def dfdk(k, fk):
    """Differentiation in Fourier space. This can be used for all functions.

    Parameters
    ----------
    k : array
        Fourier scales.
    fk : complex
        Fourier mode amplitudes.

    Returns
    -------
    dfk : complex
        Differential of the fourier modes.
    """
    dfk = 1j*k*fk
    return dfk


def dfdk2(k1, fk, k2=None):
    """Second order differentiation in Fourier space. This can be used for all functions.

    Parameters
    ----------
    k1 : array
        Fourier scales.
    fk : complex
        Fourier mode amplitudes.
    k2 : array
        Fourier scales from a different axis, if you wish to different first by one axis and then a second.

    Returns
    -------
    dfk2 : complex
        Differential of the fourier modes.
    """
    if k2 is None:
        dfk2 = -(k1**2.)*fk
    else:
        dfk2 = -k1*k2*fk
    return dfk2
