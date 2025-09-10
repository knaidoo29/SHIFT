import numpy as np
from numba import njit
from scipy.special import jv

from typing import Union


@njit
def get_Rnm_single(r: float, m: Union[int, int], knm: float, Nnm: float):
    """
    Returns the spherical bessel Rnm function.

    Parameters
    ----------
    r : float
        Radial coordinate.
    m : int or float
        The polar `m` mode.
    knm : float
        Fourier mode.
    Nnm : float
        Normalisation constant.

    Returns
    -------
    Rnm : float
        Spherical bessel Rnm function.
    """
    Rnm = (1.0 / np.sqrt(Nnm)) * jv(float(m), knm * r)
    return Rnm


@njit
def get_Rnm(r: np.ndarray, m: Union[int, float], knm: float, Nnm: float):
    """Returns the spherical bessel Rnm function.

    Parameters
    ----------
    r : array
        Radial coordinate.
    m : int or float
        The polar `m` mode.
    knm : float
        Fourier mode.
    Nnm : float
        Normalisation constant.

    Returns
    -------
    Rnm : array
        Spherical bessel Rnm function.
    """
    Rnm = np.zeros(len(r), dtype=np.float64)
    for i in range(len(r)):
        Rnm[i] = get_Rnm_single(r[i], m, knm, Nnm)
    return Rnm


@njit
def forward_half_pft(
    r: np.ndarray,
    Pm_real: np.ndarray,
    Pm_imag: np.ndarray,
    knm_flat: np.ndarray,
    Nnm_flat: np.ndarray,
    m2d_flat: np.ndarray,
    lenr: int,
    lenp: int,
):
    """
    Forward half PFT transform.

    Parameters
    ----------
    r : array
        Radial coordinates.
    pm_real : array
        Half transformed real components (i.e. forward FFT for phi direction).
    pm_imag : array
        Half transformed imaginary components (i.e. forward FFT for phi direction).
    knm_flat : array
        Fourier modes for the PFT.
    Nnm_flat : array
        Normalisation constant.
    m2d_flat : array
        The polar `m` mode.

    Returns
    -------
    Pnm : array
        PFT transformed coefficients.
    """
    dr = r[1] - r[0]
    Pnm = np.zeros(2 * lenr * lenp, dtype=np.float64)

    for i in range(lenr * lenp):
        m = m2d_flat[i]
        knm = knm_flat[i]
        Nnm = Nnm_flat[i]
        Rnm = get_Rnm(r, m, knm, Nnm)

        m_index = lenp + m if m < 0 else m

        for j in range(lenr):
            pm_index = m_index * lenr + j
            Pnm[2 * i] += r[j] * Rnm[j] * Pm_real[pm_index] * dr
            Pnm[2 * i + 1] += r[j] * Rnm[j] * Pm_imag[pm_index] * dr

    return Pnm


@njit
def backward_half_pft(
    r: np.ndarray,
    Pnm_real: np.ndarray,
    Pnm_imag: np.ndarray,
    knm_flat: np.ndarray,
    Nnm_flat: np.ndarray,
    m2d_flat: np.ndarray,
    n2d_flat: np.ndarray,
    lenr: int,
    lenp: int,
):
    """
    Forward half PFT transform.

    Parameters
    ----------
    r : array
        Radial coordinates.
    Pnm_real : array
        Real components of the PFT coefficients.
    Pnm_imag : array
        Imaginary components of the PFT coefficients.
    knm_flat : array
        Fourier modes for the PFT.
    Nnm_flat : array
        Normalisation constant.
    m2d_flat : array
        The polar `m` mode.

    Returns
    -------
    Pnm : array
        Half backwards transformed components (i.e. in radial direction).
    """
    Pm = np.zeros(2 * lenr * lenp, dtype=np.float64)

    for i in range(lenr * lenp):
        m = m2d_flat[i]
        knm = knm_flat[i]
        Nnm = Nnm_flat[i]
        Rnm = get_Rnm(r, m, knm, Nnm)

        m_index = lenp + m if m < 0 else m
        n_index = n2d_flat[i] - 1

        for j in range(lenr):
            pm_index = m_index * lenr + j
            pnm_index = m_index * lenr + n_index
            Pm[2 * pm_index] += Rnm[j] * Pnm_real[pnm_index]
            Pm[2 * pm_index + 1] += Rnm[j] * Pnm_imag[pnm_index]

    return Pm
