import numpy as np


def forward_mpi_fft_2D(f_real, f_shape, boxsize, ngrid, FFT):
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


def backward_mpi_fft_2D(f_fourier, f_shape, boxsize, ngrid, FFT):
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
    f_real = np.zeros(f_shape)
    f_real = FFT.backward(f_fourier, f_real, normalize=True)
    f_real /= (dx/np.sqrt(2.*np.pi))**2.
    f_real = f_real.real
    return f_real


def forward_mpi_fft_3D(f_real, f_shape, boxsize, ngrid, FFT):
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


def backward_mpi_fft_3D(f_fourier, f_shape, boxsize, ngrid, FFT):
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
    f_real = np.zeros(f_shape)
    f_real = FFT.backward(f_fourier, f_real, normalize=True)
    f_real /= (dx/np.sqrt(2.*np.pi))**3.
    f_real = f_real.real
    return f_real
