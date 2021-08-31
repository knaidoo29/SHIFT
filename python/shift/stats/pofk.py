import numpy as np
from knpy import bins

from .. import cart


def get_pofk_2D(dgrid, boxsize, ngrid, kmin=None, kmax=None, ncpu=None):
    """Returns the power spectrum of a 2D data set.

    Parameters
    ----------
    density : 2darray
        Density contrast.
    boxsize : float
        Box size.
    ngrid : int
        Grid divisions across one axis.
    kmin : float, optional
        Minimum Fourier mode, default = Minimum k mode of the grid.
    kmax : float, optional
        Maximum Fourier mode, default = Maximum k mode of the grid.
    ncpu : int, optional
        Number of cores for FFT.

    Returns
    -------
    k : array
        k modes for the measured power spectrum.
    keff : array
        Effective k modes for the measured power spectrum.
    pk : array
        Measure power spectrum.
    """
    kx2d, ky2d = cart.get_fourier_grid_2D(boxsize, ngrid)
    kmag = np.sqrt(kx2d**2. + ky2d**2.)
    dkgrid = cart.forward_fft_2D(dgrid, boxsize, ncpu=ncpu)
    if kmin is None:
        kmin = cart.get_kf(boxsize)
    if kmax is None:
        kmax = np.sqrt(2.)*cart.get_kn(boxsize, ngrid)
    kedges = np.linspace(kmin, kmax, int((kmax-kmin)/kmin)+1)
    k = 0.5 * (kedges[1:] + kedges[:-1])
    dk = kedges[1] - kedges[0]
    pk = np.zeros(len(k))
    keff = np.zeros(len(k))
    kf = cart.get_kf(boxsize)
    kmag = kmag.flatten()
    dkgrid = dkgrid.flatten()
    k_index = (kmag - kmin)/dk
    k_index = np.floor(k_index).astype(int)
    # cut data outside of range
    numk = len(k)
    condition = np.where((k_index >= 0) & (k_index < numk))[0]
    k_index = k_index[condition]
    kvals = kmag[condition]
    delta2 = dkgrid.real[condition]**2. + dkgrid.imag[condition]**2.
    counts = np.zeros(numk)
    counts = bins.bin_by_index(k_index, counts)
    pk = np.zeros(numk)
    keff = np.zeros(numk)
    pk = bins.bin_by_index(k_index, pk, weights=delta2)
    keff = bins.bin_by_index(k_index, keff, weights=delta2*kvals) / pk
    pk *= ((2*np.pi/boxsize)**2.)/counts
    return k, keff, pk


def get_pofk_3D(dgrid, boxsize, ngrid, kmin=None, kmax=None, ncpu=None):
    """Returns the power spectrum of a 3D data set.

    Parameters
    ----------
    density : 3darray
        Density contrast.
    boxsize : float
        Box size.
    ngrid : int
        Grid divisions across one axis.
    kmin : float, optional
        Minimum Fourier mode, default = Minimum k mode of the grid.
    kmax : float, optional
        Maximum Fourier mode, default = Maximum k mode of the grid.
    ncpu : int, optional
        Number of cores for FFT.

    Returns
    -------
    k : array
        k modes for the measured power spectrum.
    keff : array
        Effective k modes for the measured power spectrum.
    pk : array
        Measure power spectrum.
    """
    kx3d, ky3d, kz3d = cart.get_fourier_grid_3D(boxsize, ngrid)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    dkgrid = cart.forward_fft_3D(dgrid, boxsize, ncpu=ncpu)
    if kmin is None:
        kmin = cart.get_kf(boxsize)
    if kmax is None:
        kmax = np.sqrt(3.)*cart.get_kn(boxsize, ngrid)
    kedges = np.linspace(kmin, kmax, int((kmax-kmin)/kmin)+1)
    k = 0.5 * (kedges[1:] + kedges[:-1])
    dk = kedges[1] - kedges[0]
    pk = np.zeros(len(k))
    keff = np.zeros(len(k))
    kf = cart.get_kf(boxsize)
    kmag = kmag.flatten()
    dkgrid = dkgrid.flatten()
    k_index = (kmag - kmin)/dk
    k_index = np.floor(k_index).astype(int)
    # cut data outside of range
    numk = len(k)
    condition = np.where((k_index >= 0) & (k_index < numk))[0]
    k_index = k_index[condition]
    kvals = kmag[condition]
    delta2 = dkgrid.real[condition]**2. + dkgrid.imag[condition]**2.
    counts = np.zeros(numk)
    counts = bins.bin_by_index(k_index, counts)
    pk = np.zeros(numk)
    keff = np.zeros(numk)
    pk = bins.bin_by_index(k_index, pk, weights=delta2)
    keff = bins.bin_by_index(k_index, keff, weights=delta2*kvals) / pk
    pk *= ((2*np.pi/boxsize)**3.)/counts
    return k, keff, pk
