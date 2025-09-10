import numpy as np
from scipy.interpolate import interp1d

from typing import Union

from . import fft
from . import kgrid


def mult_fk_2D(
    fgrid: np.ndarray, boxsize: Union[float, list], k: np.ndarray, fk: np.ndarray
) -> np.ndarray:
    """
    Multiply 2D grid in Fourier space by k dependent function.

    Parameters
    ----------
    fgrid : 2darray
        2D grid data.
    boxsize : float
        Size of the box the grid is defined on.
    k : array
        K values for k-dependent function f(k).
    fk : array
        Function values at k.
    """
    # Creating f(k) interpolator
    xngrid = len(fgrid)
    yngrid = len(fgrid[0])
    ngrid = [xngrid, yngrid]
    fk_interpolator = interp1d(k, fk, kind="cubic")
    # Build K-grid
    kx2d, ky2d = kgrid.kgrid2D(boxsize, ngrid)
    kmag = np.sqrt(kx2d**2.0 + ky2d**2.0)
    kmag = kmag.flatten()
    # Check interpolation range
    if kmag[1:].min() > k.min():
        interpcheck1 = True
    else:
        interpcheck1 = False
    if kmag[1:].max() < k.max():
        interpcheck2 = True
    else:
        interpcheck2 = False
    assert (
        interpcheck1 and interpcheck2
    ), "ERROR! : Grid k goes beyond interpolation range"
    # Forward FFT
    fkgrid = fft.fft2D(fgrid, boxsize)
    fkgrid = fkgrid.flatten()
    # Multiply in k-space
    if k.min() != 0.0:
        fkgrid[1:] *= fk_interpolator(kmag[1:])
    else:
        fkgrid *= fk_interpolator(kmag)
    # Backward FFT
    fgrid = fft.ifft2D(fkgrid.reshape(xngrid, yngrid), boxsize).real
    return fgrid


def mult_fk_3D(
    fgrid: np.ndarray, boxsize: Union[float, list], k: np.ndarray, fk: np.ndarray
) -> np.ndarray:
    """
    Multiply 3D grid in Fourier space by k dependent function.

    Parameters
    ----------
    fgrid : ndarray
        3D grid data.
    boxsize : float
        Size of the box the grid is defined on.
    k : array
        K values for k-dependent function f(k).
    fk : array
        Function values at k.
    """
    # Creating f(k) interpolator
    xngrid = len(fgrid)
    yngrid = len(fgrid[0])
    zngrid = len(fgrid[0][0])
    ngrid = [xngrid, yngrid, zngrid]
    fk_interpolator = interp1d(k, fk, kind="cubic")
    # Build K-grid
    kx3d, ky3d, kz3d = kgrid.kgrid3D(boxsize, ngrid)
    kmag = np.sqrt(kx3d**2.0 + ky3d**2.0 + kz3d**2.0)
    kmag = kmag.flatten()
    # Check interpolation range
    if kmag[1:].min() > k.min():
        interpcheck1 = True
    else:
        interpcheck1 = False
    if kmag[1:].max() < k.max():
        interpcheck2 = True
    else:
        interpcheck2 = False
    assert (
        interpcheck1 and interpcheck2
    ), "ERROR! : Grid k goes beyond interpolation range"
    # Forward FFT
    fkgrid = fft.fft3D(fgrid, boxsize)
    fkgrid = fkgrid.flatten()
    # Multiply in k-space
    if k.min() != 0.0:
        fkgrid[1:] *= fk_interpolator(kmag[1:])
    else:
        fkgrid *= fk_interpolator(kmag)
    # Backward FFT
    fgrid = fft.ifft3D(fkgrid.reshape(xngrid, yngrid, zngrid), boxsize).real
    return fgrid
