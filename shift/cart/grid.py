import numpy as np

from typing import Tuple


def grid1D(
    boxsize: float, ngrid: int, origin: float = 0.0, dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the x coordinates of a cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.
    origin : float, optional
        Start point of the grid.
    dtype : np.dtype, optional
        Data type of the output arrays.

    Returns
    -------
    xedges : array
        x coordinate bin edges.
    x : array
        X coordinates bin centers.
    """
    xedges = np.linspace(0.0, boxsize, ngrid + 1, dtype=dtype) + origin
    x = 0.5 * (xedges[1:] + xedges[:-1])
    return xedges, x


def grid2D(
    boxsize: float, ngrid: int, origin: float = 0.0, dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the x, y coordinates of a cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size or list of length along each axis.
    ngrid : int
        Grid division along one axis or a list of the grid divisions along each
        axis.
    origin : float, optional
        Origin of the grid. If all axes begin at the same origin this can be a scalar,
        if you instead wish to specify different origins for each axis this should
        be added as a list.
    dtype : np.dtype, optional
        Data type of the output arrays.

    Returns
    -------
    x2D : array
        X coordinates on a 2D cartesian grid.
    y2D : array
        Y coordinates on a 2D cartesian grid.
    """
    if np.isscalar(boxsize) is True:
        boxsizes = [boxsize, boxsize]
    else:
        boxsizes = boxsize
    if np.isscalar(ngrid) is True:
        ngrids = [ngrid, ngrid]
    else:
        ngrids = ngrid
    if np.isscalar(origin) is True:
        origins = [origin, origin]
    else:
        origins = origin
    _, x = grid1D(boxsizes[0], ngrids[0], origin=origins[0], dtype=dtype)
    _, y = grid1D(boxsizes[1], ngrids[1], origin=origins[1], dtype=dtype)
    x2D, y2D = np.meshgrid(x, y, indexing="ij")
    return x2D, y2D


def grid3D(
    boxsize: float, ngrid: int, origin: float = 0.0, dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the x, y, z coordinates of a cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size or list of length along each axis.
    ngrid : int
        Grid division along one axis or a list of the grid divisions along each
        axis.
    origin : float, optional
        Origin of the grid. If all axes begin at the same origin this can be a scalar,
        if you instead wish to specify different origins for each axis this should
        be added as a list.
    dtype : np.dtype, optional
        Data type of the output arrays.

    Returns
    -------
    x3D : array
        X coordinates on a 3D cartesian grid.
    y3D : array
        Y coordinates on a 3D cartesian grid.
    z3D : array
        Z coordinates on a 3D cartesian grid.
    """
    if np.isscalar(boxsize) is True:
        boxsizes = [boxsize, boxsize, boxsize]
    else:
        boxsizes = boxsize
    if np.isscalar(ngrid) is True:
        ngrids = [ngrid, ngrid, ngrid]
    else:
        ngrids = ngrid
    if np.isscalar(origin) is True:
        origins = [origin, origin, origin]
    else:
        origins = origin
    _, x = grid1D(boxsizes[0], ngrids[0], origin=origins[0], dtype=dtype)
    _, y = grid1D(boxsizes[1], ngrids[1], origin=origins[1], dtype=dtype)
    _, z = grid1D(boxsizes[2], ngrids[2], origin=origins[2], dtype=dtype)
    x3D, y3D, z3D = np.meshgrid(x, y, z, indexing="ij")
    return x3D, y3D, z3D
