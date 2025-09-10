import numpy as np

from typing import Tuple, Union

from . import utils


def kgrid1D(boxsize: float, ngrid: int) -> np.ndarray:
    """
    Returns the fourier modes for the Fourier transform of a cartesian grid.

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
    k = np.arange(0.0, ngrid, 1.0)
    condition = np.where(k >= ngrid / 2.0)[0]
    k[condition] -= ngrid
    k *= kf
    return k


def kgrid2D(
    boxsize: Union[float, list], ngrid: Union[int, list]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the fourier modes for the Fourier transform of a cartesian grid.

    Parameters
    ----------
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.

    Returns
    -------
    kx2D : array
        Fourier x-mode.
    ky2D : array
        Fourier y-mode.
    """
    if np.isscalar(boxsize):
        xboxsize = boxsize
        yboxsize = boxsize
    else:
        assert (
            len(boxsize) == 2
        ), "Length of list of box dimensions must be equal to the dimenions 2."
        xboxsize = boxsize[0]
        yboxsize = boxsize[1]
    if np.isscalar(ngrid):
        xngrid = ngrid
        yngrid = ngrid
    else:
        assert (
            len(ngrid) == 2
        ), "Length of list of grid dimensions must be equal to the dimenions 2."
        xngrid = ngrid[0]
        yngrid = ngrid[1]
    kx = kgrid1D(xboxsize, xngrid)
    ky = kgrid1D(yboxsize, yngrid)
    # Create Fourier grid
    kx2D, ky2D = np.meshgrid(kx, ky, indexing="ij")
    return kx2D, ky2D


def kgrid3D(
    boxsize: Union[float, list], ngrid: Union[int, list]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the fourier modes for the Fourier transform of a cartesian grid.

    Parameters
    ----------
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list of divisions across each axes.

    Returns
    -------
    kx3D : array
        Fourier x-mode.
    ky3D : array
        Fourier y-mode.
    kz3D : array
        Fourier z-mode.
    """
    if np.isscalar(boxsize):
        xboxsize = boxsize
        yboxsize = boxsize
        zboxsize = boxsize
    else:
        assert (
            len(boxsize) == 3
        ), "Length of list of box dimensions must be equal to the dimenions 3."
        xboxsize = boxsize[0]
        yboxsize = boxsize[1]
        zboxsize = boxsize[2]
    if np.isscalar(ngrid):
        xngrid = ngrid
        yngrid = ngrid
        zngrid = ngrid
    else:
        assert (
            len(ngrid) == 3
        ), "Length of list of grid dimensions must be equal to the dimenions 3."
        xngrid = ngrid[0]
        yngrid = ngrid[1]
        zngrid = ngrid[2]
    kx = kgrid1D(xboxsize, xngrid)
    ky = kgrid1D(yboxsize, yngrid)
    kz = kgrid1D(zboxsize, zngrid)
    # Create Fourier grid
    kx3D, ky3D, kz3D = np.meshgrid(kx, ky, kz, indexing="ij")
    return kx3D, ky3D, kz3D


def kgrid1D_dct(boxsize: float, ngrid: int) -> np.ndarray:
    """
    Returns the fourier modes for the Discrete Cosine transform on a cartesian grid.

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
    k = np.arange(0.0, ngrid, 1.0)
    k *= kf / 2.0
    return k


def kgrid2D_dct(
    boxsize: Union[float, list], ngrid: Union[int, list]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the fourier modes for the Discrete Cosine transform on a 2D cartesian grid.

    Parameters
    ----------
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list of divisions across each axes.

    Returns
    -------
    kx2D : array
        Fourier x-mode.
    ky2D : array
        Fourier y-mode.
    """
    if np.isscalar(boxsize):
        xboxsize = boxsize
        yboxsize = boxsize
    else:
        assert (
            len(boxsize) == 2
        ), "Length of list of box dimensions must be equal to the dimenions 2."
        xboxsize = boxsize[0]
        yboxsize = boxsize[1]
    if np.isscalar(ngrid):
        xngrid = ngrid
        yngrid = ngrid
    else:
        assert (
            len(ngrid) == 2
        ), "Length of list of grid dimensions must be equal to the dimenions 2."
        xngrid = ngrid[0]
        yngrid = ngrid[1]
    kx = kgrid1D_dct(xboxsize, xngrid)
    ky = kgrid1D_dct(yboxsize, yngrid)
    # Create Fourier grid
    kx2D, ky2D = np.meshgrid(kx, ky, indexing="ij")
    return kx2D, ky2D


def kgrid3D_dct(
    boxsize: Union[float, list], ngrid: Union[int, list]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the fourier modes for the Discrete Cosine transform on a 3D cartesian grid.

    Parameters
    ----------
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list of divisions across each axes.

    Returns
    -------
    kx3D : array
        Fourier x-mode.
    ky3D : array
        Fourier y-mode.
    kz3D : array
        Fourier z-mode.
    """
    if np.isscalar(boxsize):
        xboxsize = boxsize
        yboxsize = boxsize
        zboxsize = boxsize
    else:
        assert (
            len(boxsize) == 3
        ), "Length of list of box dimensions must be equal to the dimenions 3."
        xboxsize = boxsize[0]
        yboxsize = boxsize[1]
        zboxsize = boxsize[2]
    if np.isscalar(ngrid):
        xngrid = ngrid
        yngrid = ngrid
        zngrid = ngrid
    else:
        assert (
            len(ngrid) == 3
        ), "Length of list of grid dimensions must be equal to the dimenions 3."
        xngrid = ngrid[0]
        yngrid = ngrid[1]
        zngrid = ngrid[2]
    kx = kgrid1D_dct(xboxsize, xngrid)
    ky = kgrid1D_dct(yboxsize, yngrid)
    kz = kgrid1D_dct(zboxsize, zngrid)
    # Create Fourier grid
    kx3D, ky3D, kz3D = np.meshgrid(kx, ky, kz, indexing="ij")
    return kx3D, ky3D, kz3D


def kgrid1D_dst(boxsize: float, ngrid: int) -> np.ndarray:
    """
    Returns the fourier modes for the Discrete Sine Transform of a cartesian grid.

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
    k = np.arange(0.0, ngrid, 1.0) + 1
    k *= kf / 2.0
    return k


def kgrid2D_dst(
    boxsize: Union[float, list], ngrid: Union[int, list]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the fourier modes for the Discrete Sine transform on a 2D cartesian grid.

    Parameters
    ----------
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list of divisions across each axes.

    Returns
    -------
    kx2D : array
        Fourier x-mode.
    ky2D : array
        Fourier y-mode.
    """
    if np.isscalar(boxsize):
        xboxsize = boxsize
        yboxsize = boxsize
    else:
        assert (
            len(boxsize) == 2
        ), "Length of list of box dimensions must be equal to the dimenions 2."
        xboxsize = boxsize[0]
        yboxsize = boxsize[1]
    if np.isscalar(ngrid):
        xngrid = ngrid
        yngrid = ngrid
    else:
        assert (
            len(ngrid) == 2
        ), "Length of list of grid dimensions must be equal to the dimenions 2."
        xngrid = ngrid[0]
        yngrid = ngrid[1]
    kx = kgrid1D_dst(xboxsize, xngrid)
    ky = kgrid1D_dst(yboxsize, yngrid)
    # Create Fourier grid
    kx2D, ky2D = np.meshgrid(kx, ky, indexing="ij")
    return kx2D, ky2D


def kgrid3D_dst(
    boxsize: Union[float, list], ngrid: Union[int, list]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the fourier modes for the Discrete Sine transform on a 3D cartesian grid.

    Parameters
    ----------
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list of divisions across each axes.

    Returns
    -------
    kx3D : array
        Fourier x-mode.
    ky3D : array
        Fourier y-mode.
    kz3D : array
        Fourier z-mode.
    """
    if np.isscalar(boxsize):
        xboxsize = boxsize
        yboxsize = boxsize
        zboxsize = boxsize
    else:
        assert (
            len(boxsize) == 3
        ), "Length of list of box dimensions must be equal to the dimenions 3."
        xboxsize = boxsize[0]
        yboxsize = boxsize[1]
        zboxsize = boxsize[2]
    if np.isscalar(ngrid):
        xngrid = ngrid
        yngrid = ngrid
        zngrid = ngrid
    else:
        assert (
            len(ngrid) == 3
        ), "Length of list of grid dimensions must be equal to the dimenions 3."
        xngrid = ngrid[0]
        yngrid = ngrid[1]
        zngrid = ngrid[2]
    kx = kgrid1D_dst(xboxsize, xngrid)
    ky = kgrid1D_dst(yboxsize, yngrid)
    kz = kgrid1D_dst(zboxsize, zngrid)
    # Create Fourier grid
    kx3D, ky3D, kz3D = np.meshgrid(kx, ky, kz, indexing="ij")
    return kx3D, ky3D, kz3D
