import numpy as np
import scipy.fft as scfft


def fft1D(f_real, boxsize, axis=-1):
    """Performs Forward FFT on real space data.

    Parameters
    ----------
    f_real : ndarray
        Real space data.
    boxsize : float
        Box size.
    axis : int, optional
        Axis to perform FFT.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    dx = boxsize / float(len(f_real))
    f_fourier = scfft.fft(f_real, axis=axis)
    f_fourier *= (dx/np.sqrt(2.*np.pi))
    return f_fourier


def ifft1D(f_fourier, boxsize, axis=-1):
    """Performs backward fft of a Fourier grid.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float
        Box size.
    axis : int, optional
        Axis to perform FFT.

    Returns
    -------
    f_real : ndarray
        Real space data.
    """
    dx = boxsize / float(len(f_fourier))
    f_real = scfft.ifft(f_fourier, axis=axis)
    f_real *= (np.sqrt(2.*np.pi)/dx)
    return f_real.real


def fft2D(f_real, boxsize):
    """Performs Forward FFT on input grid data.

    Parameters
    ----------
    f_real : 2darray
        Real space data.
    boxsize : float
        Box size.

    Returns
    -------
    f_fourier : 2darray
        Fourier modes.
    """
    assert f_real.ndim == 2, "Data must be 2 dimensional"
    dx = boxsize / float(len(f_real))
    f_fourier = scfft.fftn(f_real)
    f_fourier *= (dx/np.sqrt(2.*np.pi))**2.
    return f_fourier


def ifft2D(f_fourier, boxsize):
    """Performs backward fft of a Fourier grid.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float
        Box size.

    Returns
    -------
    f_real : 2darray
        Real space data.
    """
    assert f_fourier.ndim == 2, "Data must be 2 dimensional"
    dx = boxsize / float(len(f_fourier))
    f_real = scfft.ifftn(f_fourier)
    f_real *= (np.sqrt(2.*np.pi)/dx)**2.
    return f_real.real


def fft3D(f_real, boxsize):
    """Performs Forward FFT on input grid data.

    Parameters
    ----------
    f_real : 3darray
        Real space data.
    boxsize : float
        Box size.

    Returns
    -------
    f_fourier : 3darray
        Fourier modes.
    """
    assert f_real.ndim == 3, "Data must be 3 dimensional"
    dx = boxsize / float(len(f_real))
    f_fourier = scfft.fftn(f_real)
    f_fourier *= (dx/np.sqrt(2.*np.pi))**3.
    return f_fourier


def ifft3D(f_fourier, boxsize):
    """Performs backward fft of a Fourier grid.

    Parameters
    ----------
    f_fourier : 3darray
        Fourier modes.
    boxsize : float
        Box size.

    Returns
    -------
    f_real : 3darray
        Real space data.
    """
    assert f_fourier.ndim == 3, "Data must be 3 dimensional"
    dx = boxsize / float(len(f_fourier))
    f_real = scfft.ifftn(f_fourier)
    f_real *= (np.sqrt(2.*np.pi)/dx)**3.
    return f_real.real


def dct1D(f_real, boxsize, axis=-1, type=2):
    """Performs forward DCT in 1D.

    Parameters
    ----------
    f_real : 1darray or ndarray
        Real space data.
    Boxsize
        Box size.
    axis : int, optional
        Axis to perform DCT.
    type : int, optional
        Type of DCT for scipy to perform.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    dx = boxsize / float(len(f_real))
    f_fourier = scfft.dct(f_real, axis=axis, type=type)
    f_fourier *= (dx/np.sqrt(2.*np.pi))
    return f_fourier


def idct1D(f_fourier, boxsize, axis=-1, type=2):
    """Performs backward DCT in 1D.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float
        Box size.
    axis : int, optional
        Axis to perform DCT.
    type : int, optional
        Type of iDCT for scipy to perform.

    Returns
    -------
    f_real : 1darray or ndarray
        Real space data.
    """
    dx = boxsize / float(len(f_fourier))
    f_real = scfft.idct(f_fourier, axis=axis, type=type)
    f_real *= (np.sqrt(2.*np.pi)/dx)
    return f_real


def dct2D(f_real, boxsize, type=2):
    """Performs forward DCT in 2D.

    Parameters
    ----------
    f_real : 2darray
        Real space data.
    boxsize : float
        Box size.
    type : int, optional
        Type of DCT for scipy to perform.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    assert f_real.ndim == 2, "Data must be 2 dimensional"
    dx = boxsize / float(len(f_real))
    f_fourier = scfft.dctn(f_real, type=type)
    f_fourier *= (dx/np.sqrt(2.*np.pi))**2.
    return f_fourier


def idct2D(f_fourier, boxsize, type=-2):
    """Performs backward DCT in 2D.

    Parameters
    ----------
    f_fourier : 2darray
        Fourier modes.
    boxsize : float
        Box size.
    type : int, optional
        Type of iDCT for scipy to perform.

    Returns
    -------
    f_real : 2darray
        Real space data.
    """
    assert f_fourier.ndim == 2, "Data must be 2 dimensional"
    dx = boxsize / float(len(f_fourier))
    f_real = scfft.idctn(f_fourier, type=type)
    f_real *= (np.sqrt(2.*np.pi)/dx)**2.
    return f_real


def dct3D(f_real, boxsize, type=2):
    """Performs forward DCT in 3D.

    Parameters
    ----------
    f_real : 3darray
        Real space data.
    boxsize : float
        Box size.
    type : int, optional
        Type of DCT for scipy to perform.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    assert f_real.ndim == 3, "Data must be 3 dimensional"
    dx = boxsize / float(len(f_real))
    f_fourier = scfft.dctn(f_real, type=type)
    f_fourier *= (dx/np.sqrt(2.*np.pi))**3.
    return f_fourier


def idct3D(f_fourier, boxsize, type=2):
    """Performs backward DCT in 3D.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float
        Box size.
    type : int, optional
        Type of iDCT for scipy to perform.

    Returns
    -------
    f_real : 3darray
        Real space data.
    """
    assert f_fourier.ndim == 3, "Data must be 3 dimensional"
    dx = boxsize / float(len(f_fourier))
    f_real = scfft.idctn(f_fourier, type=type)
    f_real *= (np.sqrt(2.*np.pi)/dx)**3.
    return f_real


def dst1D(f_real, boxsize, axis=-1, type=2):
    """Performs forward DST in 1D.

    Parameters
    ----------
    f_real : 1darray or ndarray
        Real space data.
    Boxsize
        Box size.
    axis : int, optional
        Axis to perform DST.
    type : int, optional
        Type of DST for scipy to perform.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    dx = boxsize / float(len(f_real))
    f_fourier = scfft.dst(f_real, axis=axis, type=type)
    f_fourier *= (dx/np.sqrt(2.*np.pi))
    return f_fourier


def idst1D(f_fourier, boxsize, axis=-1, type=2):
    """Performs backward DST in 1D.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float
        Box size.
    axis : int, optional
        Axis to perform DST.
    type : int, optional
        Type of iDST for scipy to perform.

    Returns
    -------
    f_real : 1darray or ndarray
        Real space data.
    """
    dx = boxsize / float(len(f_fourier))
    f_real = scfft.idst(f_fourier, axis=axis, type=type)
    f_real *= (np.sqrt(2.*np.pi)/dx)
    return f_real


def dst2D(f_real, boxsize, type=2):
    """Performs forward DST in 2D.

    Parameters
    ----------
    f_real : 2darray
        Real space data.
    boxsize : float
        Box size.
    type : int, optional
        Type of DST for scipy to perform.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    assert f_real.ndim == 2, "Data must be 2 dimensional"
    dx = boxsize / float(len(f_real))
    f_fourier = scfft.dstn(f_real, type=type)
    f_fourier *= (dx/np.sqrt(2.*np.pi))**2.
    return f_fourier


def idst2D(f_fourier, boxsize, type=2):
    """Performs backward DST in 2D.

    Parameters
    ----------
    f_fourier : 2darray
        Fourier modes.
    boxsize : float
        Box size.
    type : int, optional
        Type of iDST for scipy to perform.

    Returns
    -------
    f_real : 2darray
        Real space data.
    """
    assert f_fourier.ndim == 2, "Data must be 2 dimensional"
    dx = boxsize / float(len(f_fourier))
    f_real = scfft.idstn(f_fourier, type=type)
    f_real *= (np.sqrt(2.*np.pi)/dx)**2.
    return f_real


def dst3D(f_real, boxsize, type=2):
    """Performs forward DST in 3D.

    Parameters
    ----------
    f_real : 3darray
        Real space data.
    boxsize : float
        Box size.
    type : int, optional
        Type of DST for scipy to perform.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    assert f_real.ndim == 3, "Data must be 3 dimensional"
    dx = boxsize / float(len(f_real))
    f_fourier = scfft.dstn(f_real, type=type)
    f_fourier *= (dx/np.sqrt(2.*np.pi))**3.
    return f_fourier


def idst3D(f_fourier, boxsize, type=2):
    """Performs backward DST in 3D.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float
        Box size.
    type : int, optional
        Type of iDST for scipy to perform.

    Returns
    -------
    f_real : 3darray
        Real space data.
    """
    assert f_fourier.ndim == 3, "Data must be 3 dimensional"
    dx = boxsize / float(len(f_fourier))
    f_real = scfft.idstn(f_fourier, type=type)
    f_real *= (np.sqrt(2.*np.pi)/dx)**3.
    return f_real
