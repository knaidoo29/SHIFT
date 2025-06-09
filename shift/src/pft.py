import numpy as np
from numba import njit
from scipy.special import jv


@njit
def get_rnm_single(r, m, knm, nnm):
    return (1.0 / np.sqrt(nnm)) * jv(float(m), knm * r)


@njit
def get_rnm(r, m, knm, nnm):
    rnm = np.zeros(len(r), dtype=np.float64)
    for i in range(len(r)):
        rnm[i] = get_rnm_single(r[i], m, knm, nnm)
    return rnm


@njit
def forward_half_pft(r, pm_real, pm_imag, knm_flat, nnm_flat, m2d_flat, lenr, lenp):
    dr = r[1] - r[0]
    pnm = np.zeros(2 * lenr * lenp, dtype=np.float64)

    for i in range(lenr * lenp):
        m = m2d_flat[i]
        knm = knm_flat[i]
        nnm = nnm_flat[i]
        rnm = get_rnm(r, m, knm, nnm)

        m_index = lenp + m if m < 0 else m

        for j in range(lenr):
            pm_index = m_index * lenr + j
            pnm[2 * i]     += r[j] * rnm[j] * pm_real[pm_index] * dr
            pnm[2 * i + 1] += r[j] * rnm[j] * pm_imag[pm_index] * dr

    return pnm


@njit
def backward_half_pft(r, pnm_real, pnm_imag, knm_flat, nnm_flat, m2d_flat, n2d_flat, lenr, lenp):
    dr = r[1] - r[0]
    pm = np.zeros(2 * lenr * lenp, dtype=np.float64)

    for i in range(lenr * lenp):
        m = m2d_flat[i]
        knm = knm_flat[i]
        nnm = nnm_flat[i]
        rnm = get_rnm(r, m, knm, nnm)

        m_index = lenp + m if m < 0 else m
        n_index = n2d_flat[i] - 1

        for j in range(lenr):
            pm_index = m_index * lenr + j
            pnm_index = m_index * lenr + n_index
            pm[2 * pm_index]     += rnm[j] * pnm_real[pnm_index]
            pm[2 * pm_index + 1] += rnm[j] * pnm_imag[pnm_index]

    return pm
