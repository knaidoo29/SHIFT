import numpy as np


def mpi_fft2D(f_real, f_shape, boxsize, ngrid, FFT):
    """Performs MPI Forward FFT on input grid data.

    Parameters
    ----------
    f_real : ndarray
        Input grid data, assumed to be real.
    f_shape : array
        Shape of f_real inputted.
    boxsize : float
        Box size.
    ngrid : int
        Size of the grid along one axis.
    FFT : obj
        MPI FFT object.

    Returns
    -------
    f_fourier : ndarray
        Output Fourier grid data.
    """
    dx = boxsize / float(ngrid)
    f_real = f_real + 1j*np.zeros(f_shape)
    f_fourier = FFT.forward(f_real, normalize=False)
    f_fourier *= (dx/np.sqrt(2.*np.pi))**2.
    return f_fourier


def mpi_ifft2D(f_fourier, f_shape, boxsize, ngrid, FFT):
    """Performs MPI Forward FFT on input grid data.

    Parameters
    ----------
    f_fourier : ndarray
        Output Fourier grid data.
    f_shape : array
        Shape of f_real inputted.
    boxsize : float
        Box size.
    ngrid : int
        Size of the grid along one axis.
    FFT : obj
        MPI FFT object.

    Returns
    -------
    f_real : ndarray
        Input grid data, assumed to be real.
    """
    dx = boxsize / float(ngrid)
    f = np.zeros(f_shape) + 1j*np.zeros(f_shape)
    f = FFT.backward(f_fourier, f, normalize=True)
    f /= (dx/np.sqrt(2.*np.pi))**2.
    return f.real


def mpi_fft3D(f_real, f_shape, boxsize, ngrid, FFT):
    """Performs MPI Forward FFT on input grid data.

    Parameters
    ----------
    f_real : ndarray
        Input grid data, assumed to be real.
    f_shape : array
        Shape of f_real inputted.
    boxsize : float
        Box size.
    ngrid : int
        Size of the grid along one axis.
    FFT : obj
        MPI FFT object.

    Returns
    -------
    f_fourier : ndarray
        Output Fourier grid data.
    """
    dx = boxsize / float(ngrid)
    f_real = f_real + 1j*np.zeros(f_shape)
    f_fourier = FFT.forward(f_real, normalize=False)
    f_fourier *= (dx/np.sqrt(2.*np.pi))**3.
    return f_fourier


def mpi_ifft3D(f_fourier, f_shape, boxsize, ngrid, FFT):
    """Performs MPI Forward FFT on input grid data.

    Parameters
    ----------
    f_fourier : ndarray
        Output Fourier grid data.
    f_shape : array
        Shape of f_real inputted.
    boxsize : float
        Box size.
    ngrid : int
        Size of the grid along one axis.
    FFT : obj
        MPI FFT object.

    Returns
    -------
    f : ndarray
        Input grid data.
    """
    dx = boxsize / float(ngrid)
    f = np.zeros(f_shape) + 1j*np.zeros(f_shape)
    f = FFT.backward(f_fourier, f, normalize=True)
    f /= (dx/np.sqrt(2.*np.pi))**3.
    return f.real
