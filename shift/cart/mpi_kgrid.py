import numpy as np

from typing import Tuple, Union

from . import kgrid


def mpi_kgrid1D(boxsize: float, ngrid: int, MPI: type) -> np.ndarray:
    """
    Returns the Fourier modes for the Fast Fourier Transform on a 1D
    cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.
    MPI : object
        MPIutils MPI object.

    Returns
    -------
    k : array
        Fourier modes.
    """
    k = kgrid.kgrid1D(boxsize, ngrid)
    split1, split2 = MPI.split(len(k))
    k = k[split1[MPI.rank] : split2[MPI.rank]]
    return k


def mpi_kgrid2D(
    boxsize: Union[float, list], ngrid: Union[int, list], MPI: type
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the Fourier modes for the Fast Fourier Transform on a 2D
    cartesian grid.

    Parameters
    ----------
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : object
        MPIutils MPI object.

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
    ky = mpi_kgrid1D(yboxsize, yngrid, MPI)
    kx = kgrid.kgrid1D(xboxsize, xngrid)
    # Create Fourier grid
    kx2D, ky2D = np.meshgrid(kx, ky, indexing="ij")
    return kx2D, ky2D


def mpi_kgrid3D(
    boxsize: Union[float, list], ngrid: Union[int, list], MPI: type
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the Fourier modes for the Fast Fourier Transform on a 3D
    cartesian grid.

    Parameters
    ----------
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : object
        MPIutils MPI object.

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
    ky = mpi_kgrid1D(yboxsize, yngrid, MPI)
    kx = kgrid.kgrid1D(xboxsize, xngrid)
    kz = kgrid.kgrid1D(zboxsize, zngrid)
    # Create Fourier grid
    kx3D, ky3D, kz3D = np.meshgrid(kx, ky, kz, indexing="ij")
    return kx3D, ky3D, kz3D


def mpi_kgrid1D_dct(boxsize: float, ngrid: int, MPI: type) -> np.ndarray:
    """
    Returns the Fourier modes for the Discrete Cosine Transform on a 1D
    cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.
    MPI : object
        MPIutils MPI object.

    Returns
    -------
    k : array
        Fourier modes.
    """
    k = kgrid.kgrid1D_dct(boxsize, ngrid)
    split1, split2 = MPI.split(len(k))
    k = k[split1[MPI.rank] : split2[MPI.rank]]
    return k


def mpi_kgrid2D_dct(
    boxsize: Union[float, list], ngrid: Union[int, list], MPI: type
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the Fourier modes for the Discrete Cosine Transform on a 2D
    cartesian grid.

    Parameters
    ----------
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : object
        MPIutils MPI object.

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
    ky = mpi_kgrid1D_dct(yboxsize, yngrid, MPI)
    kx = kgrid.kgrid1D_dct(xboxsize, xngrid)
    # Create Fourier grid
    kx2D, ky2D = np.meshgrid(kx, ky, indexing="ij")
    return kx2D, ky2D


def mpi_kgrid3D_dct(
    boxsize: Union[float, list], ngrid: Union[int, list], MPI: type
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the Fourier modes for the Discrete Cosine Transform on a 3D
    cartesian grid.

    Parameters
    ----------
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : object
        MPIutils MPI object.

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
    kx = mpi_kgrid1D_dct(yboxsize, yngrid, MPI)
    ky = kgrid.kgrid1D_dct(xboxsize, xngrid)
    kz = kgrid.kgrid1D_dct(zboxsize, zngrid)
    # Create Fourier grid
    kx3D, ky3D, kz3D = np.meshgrid(kx, ky, kz, indexing="ij")
    return kx3D, ky3D, kz3D


def mpi_kgrid1D_dst(boxsize: float, ngrid: int, MPI: type):
    """
    Returns the Fourier modes for the Discrete Sine Transform on a 1D
    cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.
    MPI : object
        MPIutils MPI object.

    Returns
    -------
    k : array
        Fourier modes.
    """
    k = kgrid.kgrid1D_dst(boxsize, ngrid)
    split1, split2 = MPI.split(len(k))
    k = k[split1[MPI.rank] : split2[MPI.rank]]
    return k


def mpi_kgrid2D_dst(
    boxsize: Union[float, list], ngrid: Union[int, list], MPI: type
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the Fourier modes for the Discrete Sine Transform on a 2D
    cartesian grid.

    Parameters
    ----------
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : object
        MPIutils MPI object.

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
    ky = mpi_kgrid1D_dst(yboxsize, yngrid, MPI)
    kx = kgrid.kgrid1D_dst(xboxsize, xngrid)
    # Create Fourier grid
    kx2D, ky2D = np.meshgrid(kx, ky, indexing="ij")
    return kx2D, ky2D


def mpi_kgrid3D_dst(
    boxsize: Union[float, list], ngrid: Union[int, list], MPI: type
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the Fourier modes for the Discrete Sine Transform on a 3D
    cartesian grid.

    Parameters
    ----------
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : object
        MPIutils MPI object.

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
    ky = mpi_kgrid1D_dst(yboxsize, yngrid, MPI)
    kx = kgrid.kgrid1D_dst(xboxsize, xngrid)
    kz = kgrid.kgrid1D_dst(zboxsize, zngrid)
    # Create Fourier grid
    kx3D, ky3D, kz3D = np.meshgrid(kx, ky, kz, indexing="ij")
    return kx3D, ky3D, kz3D
