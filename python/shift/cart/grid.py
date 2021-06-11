import numpy as np


def grid1d(boxsize, ngrid):
    """Returns the x coordinates of a cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.

    Returns
    -------
    xedges : array
        x coordinate bin edges.
    x : array
        X coordinates bin centers.
    """
    xedges = np.linspace(0., boxsize, ngrid + 1)
    x = 0.5*(xedges[1:] + xedges[:-1])
    return xedges, x


def grid2d(boxsize, ngrid):
    """Returns the x, y coordinates of a cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.

    Returns
    -------
    x2d : array
        X coordinates on a 2D cartesian grid.
    y2d : array
        Y coordinates on a 2D cartesian grid.
    """
    xedges, x = grid1d(boxsize, ngrid)
    x2d, y2d = np.meshgrid(x, x, indexing='ij')
    return x2d, y2d


def grid3d(boxsize, ngrid):
    """Returns the x, y, z coordinates of a cartesian grid.

    Parameters
    ----------
    boxsize : float
        Box size.
    ngrid : int
        Grid division along one axis.

    Returns
    -------
    x3d : array
        X coordinates on a 3D cartesian grid.
    y3d : array
        Y coordinates on a 3D cartesian grid.
    z3d : array
        Z coordinates on a 3D cartesian grid.
    """
    xedges, x = grid1d(boxsize, ngrid)
    x3d, y3d, z3d = np.meshgrid(x, x, x, indexing='ij')
    return x3d, y3d, z3d
