import numpy as np
from scipy import optimize
from scipy.special import jv as scipy_jv
from scipy.special import jvp as scipy_jvp
from scipy.special import jn_zeros as scipy_jn_zeros
from scipy.special import jnp_zeros as scipy_jnp_zeros

from typing import Union, Optional


def get_Jm(m: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Returns the value of Bessel functions of the first kind of order m at x.

    Paramters
    ---------
    m : int
        Order.
    x : float/array
        X-coordinates.

    Returns
    -------
    Jm : float/array
        Bessel function values.
    """
    return scipy_jv(m, x)


def get_dJm(m: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Returns the value of the derivative of Bessel functions of the first kind of order m at x.

    Paramters
    ---------
    m : int
        Order.
    x : float/array
        X-coordinates.

    Returns
    -------
    dJm : float/array
        Bessel function values.
    """
    return scipy_jvp(m, x)


def get_Jm_zeros(n: int, m: int) -> np.ndarray:
    """
    Returms the first n zeros of the Bessel function of the first kind of order m.

    Parameters
    ----------
    n : int
        Number of zeros.
    m : int
        Order.

    Returns
    -------
    xnm : array
        The zeros on the Bessel function.
    """
    xnm = scipy_jn_zeros(m, n)
    return xnm


def get_dJm_zeros(n: int, m: int) -> np.ndarray:
    """
    Returms the first n zeros of the derivative of the Bessel function of the first kind of order m.

    Parameters
    ----------
    n : int
        Number of zeros.
    m : int
        Order.

    Returns
    -------
    xmn : array
        The zeros of the derivative of the Bessel function.
    """
    xnm = scipy_jnp_zeros(m, n)
    return xnm


def get_Jm_alt(x: Union[float, np.ndarray], m: int) -> Union[float, np.ndarray]:
    """
    Wraps get_Jm in the opposite order for minimization.
    """
    return get_Jm(m, x)


def get_dJm_alt(x: Union[float, np.ndarray], m: int) -> Union[float, np.ndarray]:
    """
    Wraps get_dJm in the opposite order for minimization.
    """
    return get_dJm(m, x)


def get_Jm_large_zeros(
    m: int,
    nmax: int,
    nstop: int = 10,
    xnm2: Optional[np.ndarray] = None,
    xnm1: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns the zeros of the Bessel function. Begins by assuming
    that the zeros of the spherical Bessel function for m lie exactly between
    the zeros of the Bessel function between m and m+1. This allows us to use
    scipy's jn_zeros function. However, this function fails to return for high n.
    To work around this we estimate the first 100 zeros using scipy's jn_zero
    function and then iteratively find the roots of the next zero by assuming the
    next zero occurs pi away from the last one. Brent's method is then used to
    find a zero between pi/2 and 3pi/2 from the last zero.

    Parameters
    ----------
    m : int
        Bessel function mode.
    nmax : int
        The maximum zero found for the Bessel Function.
    nstop : int
        For n <= nstop we use scipy's jm_zeros to guess where the first nstop
        zeros are. These estimates are improved using Brent's method and assuming
        zeros lie between -pi/2 and pi/2 from the estimates.
    xnm2 : array
        Zeros for J_m-2.
    xnm1 : array
        Zeros for J_m-1.

    Returns
    -------
    xnm : array
        The zeros on the Bessel function.
    """
    if nmax <= nstop:
        nstop = nmax
    if xnm2 is None and xnm1 is None:
        xnm = get_Jm_zeros(nstop, m).tolist()
        if nstop != nmax:
            n = nstop
            while n < nmax:
                zero_last = xnm[-1]
                a = zero_last + 0.5 * np.pi
                b = zero_last + 1.5 * np.pi
                val = optimize.brentq(get_Jm_alt, a, b, args=(m))
                xnm.append(val)
                n += 1
    else:
        xnm = []
        dxnm = xnm1 - xnm2
        xnm_approx = 0.5 * ((xnm1 + dxnm) + (xnm2 + 2 * dxnm))
        for i in range(0, len(xnm_approx)):
            a = xnm_approx[i] - 0.5 * np.pi
            b = xnm_approx[i] + 0.5 * np.pi
            val = optimize.brentq(get_Jm_alt, a, b, args=(m))
            xnm.append(val)
    xnm = np.array(xnm)
    return xnm


def get_dJm_large_zeros(
    m: int,
    nmax: int,
    nstop: int = 10,
    xnm2: Optional[np.ndarray] = None,
    xnm1: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns the zeros of the Bessel function. Begins by assuming
    that the zeros of the spherical Bessel function for l lie exactly between
    the zeros of the Bessel function between m and m+1. This allows us to use
    scipy's jn_zeros function. However, this function fails to return for high n.
    To work around this we estimate the first 100 zeros using scipy's jn_zero
    function and then iteratively find the roots of the next zero by assuming the
    next zero occurs pi away from the last one. Brent's method is then used to
    find a zero between pi/2 and 3pi/2 from the last zero.

    Parameters
    ----------
    m : int
        Bessel function mode.
    nmax : int
        The maximum zero found for the Bessel Function.
    nstop : int
        For n <= nstop we use scipy's jm_zeros to guess where the first nstop
        zeros are. These estimates are improved using Brent's method and assuming
        zeros lie between -pi/2 and pi/2 from the estimates.
    xnm2 : array
        Zeros for dJ_m-2.
    xnm1 : array
        Zeros for dJ_m-1.

    Returns
    -------
    xnm : array
        The zeros on the derivative of the Bessel function.
    """
    if nmax <= nstop:
        nstop = nmax
    if xnm2 is None and xnm1 is None:
        xnm = get_dJm_zeros(nstop, m).tolist()
        if nstop != nmax:
            n = nstop
            while n < nmax:
                zero_last = xnm[-1]
                a = zero_last + 0.5 * np.pi
                b = zero_last + 1.5 * np.pi
                val = optimize.brentq(get_dJm_alt, a, b, args=(m))
                xnm.append(val)
                n += 1
    else:
        xnm = []
        dxnm = xnm1 - xnm2
        xnm_approx = 0.5 * ((xnm1 + dxnm) + (xnm2 + 2 * dxnm))
        for i in range(0, len(xnm_approx)):
            a = xnm_approx[i] - 0.5 * np.pi
            b = xnm_approx[i] + 0.5 * np.pi
            val = optimize.brentq(get_dJm_alt, a, b, args=(m))
            xnm.append(val)
    xnm = np.array(xnm)
    return xnm
