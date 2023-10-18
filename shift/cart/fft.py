import numpy as np
import scipy.fft as scfft


def fft1D(f_real, boxsize, ncpu=None, axis=None):
    """Performs Forward FFT on input grid data.

    Parameters
    ----------
    f_real : ndarray
        Input grid data.
    boxsize : float
        Box size.
    ncpu : int, optional
        Number of cpus.
    axis : int, optional
        Axis to perform FFT.

    Returns
    -------
    f_fourier : ndarray
        Output Fourier grid data.
    """
    dx = boxsize / float(len(f_real))
    if axis is None:
        f_fourier = scfft.fft(f_real, workers=ncpu)
    else:
        f_fourier = scfft.fft(f_real, workers=ncpu, axis=axis)
    f_fourier *= (dx/np.sqrt(2.*np.pi))
    return f_fourier


def ifft1D(f_fourier, boxsize, ncpu=None, axis=None):
    """Performs backward fft of a Fourier grid.

    Parameters
    ----------
    f_fourier : ndarray
        Input Fourier grid data.
    boxsize : float
        Box size.
    ncpu : int, optional
        Number of cpus.
    axis : int, optional
        Axis to perform FFT.

    Returns
    -------
    f_real : ndarray
        Output grid data.
    """
    dx = boxsize / float(len(f_fourier))
    if axis is None:
        f_real = scfft.ifft(f_fourier, workers=ncpu)
    else:
        f_real = scfft.ifft(f_fourier, workers=ncpu, axis=axis)
    f_real *= (np.sqrt(2.*np.pi)/dx)
    return f_real.real


def fft2D(f_real, boxsize, ncpu=None):
    """Performs Forward FFT on input grid data.

    Parameters
    ----------
    f_real : ndarray
        Input grid data.
    boxsize : float
        Box size.

    Returns
    -------
    f_fourier : ndarray
        Output Fourier grid data.
    """
    dx = boxsize / float(len(f_real))
    f_fourier = scfft.fftn(f_real, workers=ncpu)
    f_fourier *= (dx/np.sqrt(2.*np.pi))**2.
    return f_fourier


def ifft2D(f_fourier, boxsize, ncpu=None):
    """Performs backward fft of a Fourier grid.

    Parameters
    ----------
    f_fourier : ndarray
        Input Fourier grid data.
    boxsize : float
        Box size.

    Returns
    -------
    f_real : ndarray
        Output grid data.
    """
    dx = boxsize / float(len(f_fourier))
    f_real = scfft.ifftn(f_fourier, workers=ncpu)
    f_real *= (np.sqrt(2.*np.pi)/dx)**2.
    return f_real.real


def fft3D(f_real, boxsize, ncpu=None):
    """Performs Forward FFT on input grid data.

    Parameters
    ----------
    f_real : ndarray
        Input grid data.
    boxsize : float
        Box size.

    Returns
    -------
    f_fourier : ndarray
        Output Fourier grid data.
    """
    dx = boxsize / float(len(f_real))
    f_fourier = scfft.fftn(f_real, workers=ncpu)
    f_fourier *= (dx/np.sqrt(2.*np.pi))**3.
    return f_fourier


def ifft3D(f_fourier, boxsize, ncpu=None):
    """Performs backward fft of a Fourier grid.

    Parameters
    ----------
    f_fourier : ndarray
        Input Fourier grid data.
    boxsize : float
        Box size.

    Returns
    -------
    f_real : ndarray
        Output grid data.
    """
    dx = boxsize / float(len(f_fourier))
    f_real = scfft.ifftn(f_fourier, workers=ncpu)
    f_real *= (np.sqrt(2.*np.pi)/dx)**3.
    return f_real.real
