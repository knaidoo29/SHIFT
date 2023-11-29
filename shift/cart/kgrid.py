import numpy as np

from . import utils


def kgrid1D(boxsize, ngrid):
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


def kgrid2D(boxsize, ngrid):
    """Returns the fourier modes for the Fourier transform of a cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.

    Returns
    -------
    kx2D : array
        Fourier x-mode.
    ky2D : array
        Fourier y-mode.
    """
    k = kgrid1D(boxsize, ngrid)
    # Create Fourier grid
    kx2D, ky2D = np.meshgrid(k, k, indexing='ij')
    return kx2D, ky2D


def kgrid3D(boxsize, ngrid):
    """Returns the fourier modes for the Fourier transform of a cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.

    Returns
    -------
    kx3D : array
        Fourier x-mode.
    ky3D : array
        Fourier y-mode.
    kz3D : array
        Fourier z-mode.
    """
    k = kgrid1D(boxsize, ngrid)
    # Create Fourier grid
    kx3D, ky3D, kz3D = np.meshgrid(k, k, k, indexing='ij')
    return kx3D, ky3D, kz3D


def kgrid1D_dct(boxsize, ngrid):
    """Returns the fourier modes for the Discrete Cosine transform on a cartesian grid.

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
    kf = shift.cart.get_kf(boxsize)
    # Fourier modes along one axis
    k = np.arange(0., ngrid, 1.)
    k *= kf/2.
    return k


def kgrid2D_dct(boxsize, ngrid):
    """Returns the fourier modes for the Discrete Cosine transform on a 2D cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.

    Returns
    -------
    kx2D : array
        Fourier x-mode.
    ky2D : array
        Fourier y-mode.
    """
    k = kgrid1D_DCT(boxsize, ngrid)
    # Create Fourier grid
    kx2D, ky2D = np.meshgrid(k, k, indexing='ij')
    return kx2D, ky2D


def kgrid3D_dct(boxsize, ngrid):
    """Returns the fourier modes for the Discrete Cosine transform on a 3D cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.

    Returns
    -------
    kx3D : array
        Fourier x-mode.
    ky3D : array
        Fourier y-mode.
    kz3D : array
        Fourier z-mode.
    """
    k = kgrid1D_DCT(boxsize, ngrid)
    # Create Fourier grid
    kx3D, ky3D, kz3D = np.meshgrid(k, k, k, indexing='ij')
    return kx3D, ky3D, kz3D


def kgrid1D_dst(boxsize, ngrid):
    """Returns the fourier modes for the Discrete Sine Transform of a cartesian grid.

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
    kf = shift.cart.get_kf(boxsize)
    # Fourier modes along one axis
    k = np.arange(0., ngrid, 1.)+1
    k *= kf/2.
    return k


def kgrid2D_dst(boxsize, ngrid):
    """Returns the fourier modes for the Discrete Sine transform on a 2D cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.

    Returns
    -------
    kx2D : array
        Fourier x-mode.
    ky2D : array
        Fourier y-mode.
    """
    k = kgrid1D_DST(boxsize, ngrid)
    # Create Fourier grid
    kx2D, ky2D = np.meshgrid(k, k, indexing='ij')
    return kx2D, ky2D


def kgrid3D_dst(boxsize, ngrid):
    """Returns the fourier modes for the Discrete Sine transform on a 3D cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.

    Returns
    -------
    kx3D : array
        Fourier x-mode.
    ky3D : array
        Fourier y-mode.
    kz3D : array
        Fourier z-mode.
    """
    k = kgrid1D_DST(boxsize, ngrid)
    # Create Fourier grid
    kx3D, ky3D, kz3D = np.meshgrid(k, k, k, indexing='ij')
    return kx3D, ky3D, kz3D
