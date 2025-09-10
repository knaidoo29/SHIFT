import numpy as np

from . import bessel


def get_n(Nmax: int) -> np.ndarray:
    """Returns an array of n up to Nn.

    Parameters
    ----------
    Nmax : int
        Maximum n.

    Returns
    -------
    n : array
        N values for Bessel zeros.
    """
    n = np.arange(Nmax + 1)[1:]
    return n


def get_m(Nm: int) -> np.ndarray:
    """Returns m for FFT of the 1D angular polar components.

    Parameters
    ----------
    Nm : int
        Length of phi grid.

    Returns
    -------
    m : array
        Bessel orders.
    """
    m = np.arange(Nm)
    condition = np.where(m > float(Nm) / 2.0)[0]
    m[condition] -= Nm
    return m


def get_knm(xnm: np.ndarray, Rmax: float) -> np.ndarray:
    """
    Returns the k components given zeros provided and the maximum radius.

    Parameters
    ----------
    xnm : array
        Location of zeros.
    Rmax : float
        Maximum radius.

    Returns
    -------
    knm : array
        Corresponding Fourier mode k.
    """
    knm = xnm / Rmax
    return knm


def get_Nnm_zero(m: int, xnm: np.ndarray, Rmax: float) -> np.ndarray:
    """
    Returns the normalisation constant for zero-value boundaries.

    Parameters
    ----------
    m : int
        Order
    xnm : array
        Location of zeros for zero-value boundaries.
    Rmax : float
        Maximum radius.

    Returns
    -------
    Nnm : array
        Normalisation constants.
    """
    Nnm = ((Rmax**2.0) / 2.0) * (bessel.get_Jm(m + 1, xnm) ** 2.0)
    return Nnm


def get_Nnm_deri(m: int, xnm: np.ndarray, Rmax: float) -> np.ndarray:
    """
    Returns the normalisation constant for derivative boundaries.

    Parameters
    ----------
    m : int
        Order
    xnm : array
        Location of zeros for derivative boundaries.
    Rmax : float
        Maximum radius.

    Returns
    -------
    Nnm : array
        Normalisation constants.
    """
    Nnm = (
        ((Rmax**2.0) / 2.0)
        * (1.0 - (m**2.0) / (xnm**2.0))
        * (bessel.get_Jm(m, xnm) ** 2.0)
    )
    return Nnm


def get_Rnm(r: np.ndarray, m: int, knm: float, Nnm: float) -> np.ndarray:
    """
    Radial component of the polar basis function.

    Parameters
    ----------
    r : array
        Radial values.
    m : int
        Order
    knm : float
        Corresponding k Fourier mode for n and m.
    Nnm : float
        Corresponding normalisation constant.

    Returns
    -------
    Rnm : array
        Radial basis values.
    """
    Rnm = (1.0 / np.sqrt(Nnm)) * bessel.get_Jm(m, knm * r)
    return Rnm


def get_eix(x: np.ndarray) -> np.ndarray:
    """
    Euler's equation.

    Parameters
    ----------
    x : array
        X-coordinates.

    Returns
    -------
    eix : array
        Euler's equation.
    """
    eix = np.cos(x) + 1j * np.sin(x)
    return eix


def get_eix_star(x: np.ndarray) -> np.ndarray:
    """
    Euler's equation.

    Parameters
    ----------
    x : array
        X-coordinates.

    Returns
    -------
    eix : array
        Euler's equation.
    """
    eix = np.cos(x) - 1j * np.sin(x)
    return eix


def get_Phi_m(m: int, phi: np.ndarray) -> np.ndarray:
    """
    Angular component of the polar basis function.

    Parameters
    ----------
    m : int
        Order
    phi : array
        Angular value in radians.

    Returns
    -------
    Phi_m : array
        Angular basis function.
    """
    Phi_m = get_eix(m * phi) / np.sqrt(2.0 * np.pi)
    return Phi_m


def get_Phi_star_m(m: int, phi: np.ndarray) -> np.ndarray:
    """
    Angular component of the polar basis function.

    Parameters
    ----------
    m : int
        Order
    phi : array
        Angular value in radians.

    Returns
    -------
    Phi_m : array
        Angular basis function.
    """
    Phi_m = get_eix_star(m * phi) / np.sqrt(2.0 * np.pi)
    return Phi_m


def get_Psi_nm(
    n: int, m: int, r: np.ndarray, phi: np.ndarray, knm: float, Nnm: float
) -> np.ndarray:
    """
    Polar radial basis function

    Parameters
    ----------
    n : int
        Number of zeros.
    m : int
        Bessel order.
    r : array
        Radius.
    phi : array
        Angle.
    knm : float
        Corresponding k Fourier mode for n and m.
    Nnm : float
        Corresponding normalisation constant.

    Returns
    -------
    Psi_nm : array
        Polar radial basis function.
    """
    Phi_m = get_Phi_m(m, phi)
    Rnm = get_Rnm(r, m, knm, Nnm)
    Psi_nm = Phi_m
    Psi_nm.real *= Rnm
    Psi_nm.imag *= Rnm
    return Psi_nm


def get_Psi_star_nm(
    n: int, m: int, r: np.ndarray, phi: np.ndarray, knm: float, Nnm: float
) -> np.ndarray:
    """Polar radial basis function

    Parameters
    ----------
    n : int
        Number of zeros.
    m : int
        Bessel order.
    r : array
        Radius.
    phi : array
        Angle.
    knm : float
        Corresponding k Fourier mode for n and m.
    Nnm : float
        Corresponding normalisation constant.

    Returns
    -------
    Psi_nm : array
        Polar radial basis function.
    """
    Phi_m = get_Phi_star_m(m, phi)
    Rnm = get_Rnm(r, m, knm, Nnm)
    Psi_nm = Phi_m
    Psi_nm.real *= Rnm
    Psi_nm.imag *= Rnm
    return Psi_nm
