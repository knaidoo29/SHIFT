import numpy as np

from typing import Optional


def dfdk(k: np.ndarray, fk: np.ndarray) -> np.ndarray:
    """
    Differentiation in Fourier space. This can be used for all functions.

    Parameters
    ----------
    k : array
        Fourier scales.
    fk : complex array
        Fourier mode amplitudes.

    Returns
    -------
    dfk : complex array
        Differential of the fourier modes.
    """
    dfk = 1j * k * fk
    return dfk


def dfdk2(
    k1: np.ndarray, fk: np.ndarray, k2: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Second order differentiation in Fourier space. This can be used for all functions.

    Parameters
    ----------
    k1 : array
        Fourier scales.
    fk : complex array
        Fourier mode amplitudes.
    k2 : array
        Fourier scales from a different axis, if you wish to different first by one axis and then a second.

    Returns
    -------
    dfk2 : complex
        Differential of the fourier modes.
    """
    if k2 is None:
        dfk2 = -(k1**2.0) * fk
    else:
        dfk2 = -k1 * k2 * fk
    return dfk2
