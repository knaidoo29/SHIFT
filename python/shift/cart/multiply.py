import numpy as np
from scipy.interpolate import interp1d

from . import fft
from . import kgrid


def fourier_multiply_2D(dgrid, boxsize, k, fk, ncpu=None):
    """Multiply 2D grid in Fourier space by k dependent function.

    Parameters
    ----------
    dgrid : 2darray
        2D grid data.
    boxsize : float
        Size of the box the grid is defined on.
    k : array
        K values for k-dependent function f(k).
    fk : array
        Function values at k.
    ncpu : int, optional
        Number of CPUs for FFT.
    """
    # Creating f(k) interpolator
    ngrid = len(dgrid)
    fk_interpolator = interp1d(k, fk, kind='cubic')
    # Build K-grid
    kx2d, ky2d = kgrid.get_fourier_grid_2D(boxsize, ngrid)
    kmag = np.sqrt(kx2d**2. + ky2d**2.)
    # Check interpolation range
    if kmag[1:].min() > k.min():
        interpcheck1 = True
    else:
        interpcheck1 = False
    if kmag[1:].max() < k.max():
        interpcheck2 = True
    else:
        interpcheck2 = False
    assert interpcheck1 and interpcheck2, "ERROR! : Grid k goes beyond interpolation range"
    # Forward FFT
    dkgrid = fft.forward_fft_2D(dgrid, boxsize, ncpu=ncpu)
    dkgrid = dkgrid.flatten()
    # Multiply in k-space
    if k.min() != 0.:
        dkgrid[1:] *= fk_interpolator(kmag[1:])
    else:
        dkgrid *= fk_interpolator(kmag)
    # Backward FFT
    dgrid = fft.backward_fft_2D(dkgrid.reshape(ngrid, ngrid)).real
    return dgrid


def fourier_multiply_3D(dgrid, boxsize, k, fk, ncpu=None):
    """Multiply 3D grid in Fourier space by k dependent function.

    Parameters
    ----------
    dgrid : ndarray
        3D grid data.
    boxsize : float
        Size of the box the grid is defined on.
    k : array
        K values for k-dependent function f(k).
    fk : array
        Function values at k.
    ncpu : int, optional
        Number of CPUs for FFT.
    """
    # Creating f(k) interpolator
    ngrid = len(dgrid)
    fk_interpolator = interp1d(k, fk, kind='cubic')
    # Build K-grid
    kx3d, ky3d, kz3d = kgrid.get_fourier_grid_3D(boxsize, ngrid)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d)
    # Check interpolation range
    if kmag[1:].min() > k.min():
        interpcheck1 = True
    else:
        interpcheck1 = False
    if kmag[1:].max() < k.max():
        interpcheck2 = True
    else:
        interpcheck2 = False
    assert interpcheck1 and interpcheck2, "ERROR! : Grid k goes beyond interpolation range"
    # Forward FFT
    dkgrid = fft.forward_fft_3D(dgrid, boxsize, ncpu=ncpu)
    dkgrid = dkgrid.flatten()
    # Multiply in k-space
    if k.min() != 0.:
        dkgrid[1:] *= fk_interpolator(kmag[1:])
    else:
        dkgrid *= fk_interpolator(kmag)
    # Backward FFT
    dgrid = fft.backward_fft_3D(dkgrid.reshape(ngrid, ngrid, ngrid)).real
    return dgrid
