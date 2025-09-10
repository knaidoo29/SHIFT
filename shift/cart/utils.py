import numpy as np
from scipy import fft


def get_kf(boxsize: float) -> float:
    """
    Returns the Fundamental Fourier mode.

    Parameters
    ----------
    boxsize : float
        Box size.

    Returns
    -------
    kf : float
        Fundamental mode.
    """
    kf = 2.0 * np.pi / boxsize
    return kf


def get_kn(boxsize: float, ngrid: int) -> float:
    """
    Returns the Nyquist Fourier mode.

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
    kn = ngrid * np.pi / boxsize
    return kn


def fftshift(freq: np.ndarray) -> np.ndarray:
    """
    Centers FFT frequencies so that 0 is in the center.

    Parameters
    ----------
    freq : array
        Reorders any FFT frequency-like array to have the 0th element in the center.
    """
    return fft.fftshift(freq)


def ifftshift(freq: np.ndarray) -> np.ndarray:
    """
    Uncenters FFT frequencies so that 0 is no longer in the center.

    Parameters
    ----------
    freq : array
        Reorders any FFT frequency-like array to have the 0th element back to the
        normal convention and not in the center.
    """
    return fft.ifftshift(freq)


def normalise_freq(freq: np.ndarray, boxsize: float) -> np.ndarray:
    """
    Normalises fourier frequencies, i.e. removing the dependency on boxsize.

    Parameters
    ----------
    freq : array
        FFT frequencies.
    boxsize : float
        Size of the original box.
    """
    shape = np.shape(freq)
    grid = shape[0]
    dim = len(shape)
    dx = boxsize / grid
    freq /= dx**dim
    return freq


def unnormalise_freq(freq: np.ndarray, boxsize: float) -> np.ndarray:
    """
    Unnormalises fourier frequencies, i.e. adds the dependency on boxsize.

    Parameters
    ----------
    freq : array
        FFT frequencies.
    boxsize : float
        Size of the original box.
    """
    shape = np.shape(freq)
    grid = shape[0]
    dim = len(shape)
    dx = boxsize / grid
    freq *= dx**dim
    return freq
