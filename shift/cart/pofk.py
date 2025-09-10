import numpy as np

from typing import Optional, Tuple, Union

from . import kgrid
from . import fft
from . import utils

from .. import src


def get_pofk_2D(
    dgrid: np.ndarray,
    boxsize: Union[float, list],
    ngrid: Union[int, list],
    kmin: Optional[float] = None,
    kmax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the power spectrum of a 2D data set.

    Parameters
    ----------
    dgrid : 2darray
        Density contrast.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    kmin : float, optional
        Minimum Fourier mode, default = Minimum k mode of the grid.
    kmax : float, optional
        Maximum Fourier mode, default = Maximum k mode of the grid.

    Returns
    -------
    k : array
        k modes for the measured power spectrum.
    keff : array
        Effective k modes for the measured power spectrum.
    pk : array
        Measure power spectrum.
    """
    kx2d, ky2d = kgrid.kgrid2D(boxsize, ngrid)
    kmag = np.sqrt(kx2d**2.0 + ky2d**2.0)
    dkgrid = fft.fft2D(dgrid, boxsize)
    if kmin is None:
        kmin = utils.get_kf(boxsize)
    if kmax is None:
        kmax = np.sqrt(2.0) * utils.get_kn(boxsize, ngrid)
    kedges = np.linspace(kmin, kmax, int((kmax - kmin) / kmin) + 1)
    k = 0.5 * (kedges[1:] + kedges[:-1])
    dk = kedges[1] - kedges[0]
    pk = np.zeros(len(k))
    keff = np.zeros(len(k))
    kf = utils.get_kf(boxsize)
    kmag = kmag.flatten()
    dkgrid = dkgrid.flatten()
    k_index = (kmag - kmin) / dk
    k_index = np.floor(k_index).astype(int)
    # cut data outside of range
    numk = len(k)
    condition = np.where((k_index >= 0) & (k_index < numk))[0]
    k_index = k_index[condition]
    kvals = kmag[condition]
    delta2 = dkgrid.real[condition] ** 2.0 + dkgrid.imag[condition] ** 2.0
    counts = src.binbyindex(k_index, np.ones(len(k_index)), numk)
    pk = src.binbyindex(k_index, delta2, numk)
    keff = src.binbyindex(k_index, delta2 * kvals, numk)
    cond = np.where(pk != 0.0)[0]
    keff[cond] /= pk[cond]
    cond = np.where(pk == 0.0)[0]
    keff[cond] = np.nan
    pk *= ((2 * np.pi / boxsize) ** 2.0) / counts
    return k, keff, pk


def get_pofk_3D(
    dgrid: np.ndarray,
    boxsize: Union[float, list],
    ngrid: Union[int, list],
    kmin: Optional[float] = None,
    kmax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the power spectrum of a 3D data set.

    Parameters
    ----------
    dgrid : 3darray
        Density contrast.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    kmin : float, optional
        Minimum Fourier mode, default = Minimum k mode of the grid.
    kmax : float, optional
        Maximum Fourier mode, default = Maximum k mode of the grid.

    Returns
    -------
    k : array
        k modes for the measured power spectrum.
    keff : array
        Effective k modes for the measured power spectrum.
    pk : array
        Measure power spectrum.
    """
    kx3d, ky3d, kz3d = kgrid.kgrid3D(boxsize, ngrid)
    kmag = np.sqrt(kx3d**2.0 + ky3d**2.0 + kz3d**2.0)
    dkgrid = fft.fft3D(dgrid, boxsize)
    if kmin is None:
        kmin = utils.get_kf(boxsize)
    if kmax is None:
        kmax = np.sqrt(3.0) * utils.get_kn(boxsize, ngrid)
    kedges = np.linspace(kmin, kmax, int((kmax - kmin) / kmin) + 1)
    k = 0.5 * (kedges[1:] + kedges[:-1])
    dk = kedges[1] - kedges[0]
    pk = np.zeros(len(k))
    keff = np.zeros(len(k))
    kf = utils.get_kf(boxsize)
    kmag = kmag.flatten()
    dkgrid = dkgrid.flatten()
    k_index = (kmag - kmin) / dk
    k_index = np.floor(k_index).astype(int)
    # cut data outside of range
    numk = len(k)
    condition = np.where((k_index >= 0) & (k_index < numk))[0]
    k_index = k_index[condition]
    kvals = kmag[condition]
    delta2 = dkgrid.real[condition] ** 2.0 + dkgrid.imag[condition] ** 2.0
    counts = src.binbyindex(k_index, np.ones(len(k_index)), numk)
    pk = src.binbyindex(k_index, delta2, numk)
    keff = src.binbyindex(k_index, delta2 * kvals, numk)
    cond = np.where(pk != 0.0)[0]
    keff[cond] /= pk[cond]
    cond = np.where(pk == 0.0)[0]
    keff[cond] = np.nan
    pk *= ((2 * np.pi / boxsize) ** 3.0) / counts
    return k, keff, pk
