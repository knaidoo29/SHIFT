import numpy as np

from . import utils


def get_fourier_grid_1D(boxsize, ngrid):
    """Returns the fourier modes for the Fourier transform of a cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.

    Returns
    -------
    k : array
        Fourier modes.
    """
    # fundamental frequency
    kf = utils.get_kf(boxsize)
    # Fourier modes along one axis
    k = np.arange(0., ngrid, 1.)
    condition = np.where(k >= ngrid/2.)[0]
    k[condition] -= ngrid
    k *= kf
    return k


def get_fourier_grid_2D(boxsize, ngrid):
    """Returns the fourier modes for the Fourier transform of a cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.

    Returns
    -------
    kx2d : array
        Fourier x-mode.
    ky2d : array
        Fourier y-mode.
    """
    k = get_fourier_grid_1D(boxsize, ngrid)
    # Create Fourier grid
    kx2d, ky2d = np.meshgrid(k, k, indexing='ij')
    return kx2d, ky2d


def get_fourier_grid_3D(boxsize, ngrid):
    """Returns the fourier modes for the Fourier transform of a cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.

    Returns
    -------
    kx3d : array
        Fourier x-mode.
    ky3d : array
        Fourier y-mode.
    kz3d : array
        Fourier z-mode.
    """
    k = get_fourier_grid_1D(boxsize, ngrid)
    # Create Fourier grid
    kx3d, ky3d, kz3d = np.meshgrid(k, k, k, indexing='ij')
    return kx3d, ky3d, kz3d
