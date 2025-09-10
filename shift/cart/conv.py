import numpy as np


def convolve_gaussian(k: np.ndarray, sigma: float) -> np.ndarray:
    """
    Convolution weights in Fourier space.

    Parameters
    ----------
    k : array
        Fourier modes.
    sigma : float
        Gaussian scale.

    Returns
    -------
    kernel : array
        Convolution kernel
    """
    kernel = np.exp(-0.5 * (k * sigma) ** 2.0)
    return kernel
