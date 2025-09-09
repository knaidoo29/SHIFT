import numpy as np
import pytest
import shift

def test_zero_sigma():
    """If sigma=0, the Gaussian should return all ones regardless of k."""
    k = np.array([0.0, 1.0, 5.0])
    result = shift.cart.convolve_gaussian(k, 0.0)
    expected = np.ones_like(k)
    np.testing.assert_allclose(result, expected)

def test_zero_k():
    """If k=0, the result should always be 1 regardless of sigma."""
    k = np.array([0.0])
    result = shift.cart.convolve_gaussian(k, 2.5)
    expected = np.array([1.0])
    np.testing.assert_allclose(result, expected)

def test_known_values():
    """Check against manually computed values for known inputs."""
    k = np.array([1.0, 2.0])
    sigma = 1.0
    expected = np.exp(-0.5 * (k * sigma) ** 2)
    result = shift.cart.convolve_gaussian(k, sigma)
    np.testing.assert_allclose(result, expected)

def test_shape_preserved():
    """Ensure input and output shapes match."""
    k = np.linspace(-10, 10, 100)
    result = shift.cart.convolve_gaussian(k, 0.5)
    assert result.shape == k.shape

def test_large_sigma():
    """For very large sigma, results should approach zero for nonzero k."""
    k = np.array([1.0, 10.0, 100.0])
    result = shift.cart.convolve_gaussian(k, 1e6)
    assert np.all(result < 1e-10)

def test_negative_sigma():
    """Negative sigma should behave the same as positive (since squared)."""
    k = np.array([1.0, 2.0, 3.0])
    result_pos = shift.cart.convolve_gaussian(k, 2.0)
    result_neg = shift.cart.convolve_gaussian(k, -2.0)
    np.testing.assert_allclose(result_pos, result_neg)
