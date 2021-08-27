import numpy as np

from .. import cart


def get_pofk_2D(density, boxsize, ngrid, kmin=None, kmax=None, ncpu=None):
    """Returns the power spectrum of the input particle data in 2D.

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
    for i in range(0, len(k)):
        condition = np.where((kmag >= k[i] - 0.5*dk) & (kmag < k[i] + 0.5*dk))
        kvals = kmag[condition]
        delta2 = dkgrid.real[condition]**2. + dkgrid.imag[condition]**2.
        pk[i] = np.mean(delta2)*(2.*np.pi/boxsize)**3.
        keff[i] = np.sum(delta2 * kvals)/np.sum(delta2)
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
    for i in range(0, len(k)):
        condition = np.where((kmag >= k[i] - 0.5*dk) & (kmag < k[i] + 0.5*dk))
        kvals = kmag[condition]
        delta2 = dkgrid.real[condition]**2. + dkgrid.imag[condition]**2.
        pk[i] = np.mean(delta2)*(2.*np.pi/boxsize)**3.
        keff[i] = np.sum(delta2 * kvals)/np.sum(delta2)
    return k, keff, pk
