import numpy as np

from .. import cart


def uspheregrid(Nphi, Ntheta=None):
    """Returns a 2D longitude lattitude grid on a unit sphere using the convention,
    phi = [0, 2pi], theta=[0, phi] with theta=0 pointint to the north pole.

    Parameters
    ----------
    Nphi : int
        Number of divisions along the longitude direction, must be even.
    Ntheta : int, optional
        Number of divisions along the latitude directions.

    Returns
    -------
    r2d : array
        Radial grid.
    p2d : array
        Phi grid.
    """
    if Ntheta is None:
        assert Nphi % 2 == 0, "Nphi must be even if Ntheta is None."
        Ntheta = int(Nphi/2)
    pedges, p = cart.grid1d(2.*np.pi, Nr)
    tedges, t = cart.grid1d(2.*np.pi, Nphi)
    p2d, t2d = np.meshgrid(p, t[::-1], indexing='ij')
    return p2d, t2d
