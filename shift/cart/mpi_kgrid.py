import numpy as np

from . import kgrid
from . import utils


def mpi_kgrid1D(boxsize, ngrid, MPI):
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
    k = kgrid.kgrid1D(boxsize, ngrid)
    split1, split2 = MPI.split(len(k))
    k = k[split1[MPI.rank]:split2[MPI.rank]]
    return k


def mpi_kgrid2D(boxsize, ngrid, MPI):
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
    _k = mpi_kgrid1D(boxsize, ngrid, MPI)
    k = kgrid.kgrid1D(boxsize, ngrid)
    # Create Fourier grid
    kx2D, ky2D = np.meshgrid(k, _k, indexing='ij')
    return kx2D, ky2D


def mpi_kgrid3D(boxsize, ngrid, MPI):
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
    _k = mpi_kgrid1D(boxsize, ngrid, MPI)
    k = kgrid.kgrid1D(boxsize, ngrid)
    # Create Fourier grid
    kx3D, ky3D, kz3D = np.meshgrid(k, _k, k, indexing='ij')
    # return kx3D, ky3D, kz3D
    # kx = kgrid.kgrid1D(boxsize, ngrid)
    # ky = np.copy(kx)
    # kz = np.copy(kx)
    # Ngrids = np.array([ngrid, ngrid, ngrid], dtype='int')
    # partitionk = MPI.check_partition(Ngrids, np.array(shape))
    # kx3D, ky3D, kz3D = MPI.create_split_ndgrid([kx, ky, kz], partitionk)
    return kx3D, ky3D, kz3D
