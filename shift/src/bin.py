import numpy as np
from numba import njit


@njit
def binbyindex(ind, weights, binlength):
    """
    Bins weights according to the bin index.

    Parameters
    ----------
    ind : np.ndarray (int)
        Index of the bin for each weight.
    weights : np.ndarray (float64)
        Weights to be binned.
    binlength : int
        Length of the bin array.

    Returns
    -------
    bins : np.ndarray (float64)
        The output bins.
    """
    bins = np.zeros(binlength, dtype=np.float64)

    for i in range(ind.size):
        i0 = ind[i]
        bins[i0] += weights[i]

    return bins
