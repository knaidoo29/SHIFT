import numpy as np

from .. import cart


def polargrid(Rmax, Nr, Nphi):
    """Returns a 2D polar grid.

    Parameters
    ----------
    Rmax : float
        Maximum radius.
    Nr : int
        Number of elements along the radial axis.
    Nphi : int
        Number of elements along the angular axis.

    Returns
    -------
    r2d : array
        Radial grid.
    p2d : array
        Phi grid.
    """
    redges, r = cart.grid1D(Rmax, Nr)
    pedges, p = cart.grid1D(2.*np.pi, Nphi)
    p2d, r2d = np.meshgrid(p, r, indexing='ij')
    return p2d, r2d


def wrap_polar(f):
    """Wraps polar grid, which is useful for plotting purposes.

    Parameters
    ----------
    f : 2darray
        Field polar grid.

    Returns
    -------
    f : 2darray
        Wrapped field polar grid.
    """
    f = np.concatenate([f, np.array([f[0]])])
    return f


def unwrap_polar(f):
    """Unwraps polar grid.

    Parameters
    ----------
    f : 2darray
        Wrapped field polar grid.

    Returns
    -------
    f : 2darray
        Field polar grid.
    """
    f = f[:-1]
    return f


def wrap_phi(p2d):
    """Wraps polar grid, which is useful for plotting purposes.

    Parameters
    ----------
    p2d : array
        Phi grid.

    Returns
    -------
    p2d : array
        Wrapped Phi grid.
    """
    p2d = wrap_polar(p2d)
    p2d[-1] = 2.*np.pi
    return p2d


def unwrap_phi(f):
    """Unwraps polar grid.

    Parameters
    ----------
    p2d : array
        Wrapped Phi grid.

    Returns
    -------
    p2d : array
        Phi grid.
    """
    p2d = p2d[:-1]
    return p2d
