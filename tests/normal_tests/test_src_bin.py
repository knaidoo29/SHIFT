import numpy as np
import pytest

from shift.src import binbyindex


# -----------------
# binbyindex tests
# -----------------

def test_binbyindex_basic():
    ind = np.array([0, 1, 0, 2])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    bins = binbyindex(ind, weights, 3)
    np.testing.assert_array_equal(bins, np.array([4.0, 2.0, 4.0]))


def test_binbyindex_empty_bins():
    ind = np.array([1, 1, 1])
    weights = np.array([1.0, 2.0, 3.0])
    bins = binbyindex(ind, weights, 4)
    np.testing.assert_array_equal(bins, np.array([0.0, 6.0, 0.0, 0.0]))
