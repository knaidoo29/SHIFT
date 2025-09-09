import numpy as np
import pytest

from shift.cart import get_kf, get_kn, fftshift, ifftshift, normalise_freq, unnormalise_freq


def test_get_kf():
    box = 10.0
    kf = get_kf(box)
    expected = 2.0 * np.pi / box
    assert kf == pytest.approx(expected)


def test_get_kn():
    box = 10.0
    ngrid = 16
    kn = get_kn(box, ngrid)
    expected = ngrid * np.pi / box
    assert kn == pytest.approx(expected)


def test_fftshift_ifftshift():
    arr = np.arange(8)
    shifted = fftshift(arr)
    unshifted = ifftshift(shifted)
    np.testing.assert_array_equal(arr, unshifted)

    # Check that fftshift actually moves zero-frequency to center
    arr2 = np.array([0, 1, 2, 3])
    shifted2 = fftshift(arr2)
    np.testing.assert_array_equal(shifted2, np.array([2, 3, 0, 1]))


def test_normalise_unnormalise_freq():
    box = 1.0
    freq = np.arange(8, dtype=float)
    freq_copy = freq.copy()
    norm = normalise_freq(freq_copy, box)
    unnorm = unnormalise_freq(norm, box)
    np.testing.assert_allclose(freq, unnorm, rtol=1e-12)

    # Check that normalisation scales correctly for known case
    freq_test = np.array([1.0, 2.0, 3.0])
    norm_test = normalise_freq(freq_test.copy(), box)
    expected = freq_test / (box / len(freq_test))**len(freq_test.shape)
    np.testing.assert_allclose(norm_test, expected)
