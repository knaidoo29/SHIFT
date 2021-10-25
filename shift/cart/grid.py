import numpy as np


def grid1d(boxsize, ngrid, origin=0.):
    """Returns the x coordinates of a cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.
    origin : float, optional
        Start point of the grid.

    Returns
    -------
    xedges : array
        x coordinate bin edges.
    x : array
        X coordinates bin centers.
    """
    xedges = np.linspace(0., boxsize, ngrid + 1) + origin
    x = 0.5*(xedges[1:] + xedges[:-1])
    return xedges, x


def grid2d(boxsize, ngrid, origin=0.):
    """Returns the x, y coordinates of a cartesian grid.

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

    Returns
    -------
    x2d : array
        X coordinates on a 2D cartesian grid.
    y2d : array
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
    xedges, x = grid1d(boxsizes[0], ngrids[0], origin=origins[0])
    yedges, y = grid1d(boxsizes[1], ngrids[1], origin=origins[1])
    x2d, y2d = np.meshgrid(x, y, indexing='ij')
    return x2d, y2d


def grid3d(boxsize, ngrid, origin=0.):
    """Returns the x, y, z coordinates of a cartesian grid.

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

    Returns
    -------
    x3d : array
        X coordinates on a 3D cartesian grid.
    y3d : array
        Y coordinates on a 3D cartesian grid.
    z3d : array
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
    xedges, x = grid1d(boxsizes[0], ngrids[0], origin=origins[0])
    yedges, y = grid1d(boxsizes[1], ngrids[1], origin=origins[1])
    zedges, z = grid1d(boxsizes[2], ngrids[2], origin=origins[2])
    x3d, y3d, z3d = np.meshgrid(x, y, z, indexing='ij')
    return x3d, y3d, z3d
