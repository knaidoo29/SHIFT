import numpy as np
import pytest

from shift.cart import dfdk, dfdk2


def test_dfdk_zero_k():
    """If k=0, derivative should be zero regardless of fk."""
    k = np.array([0.0, 0.0, 0.0])
    fk = np.array([1.0, -2.0, 3.0], dtype=complex)
    result = dfdk(k, fk)
    expected = np.zeros_like(fk)
    np.testing.assert_allclose(result, expected)


def test_dfdk_known_values():
    """Check dfdk with known inputs."""
    k = np.array([1.0, 2.0])
    fk = np.array([1.0 + 0j, 2.0 + 0j])
    result = dfdk(k, fk)
    expected = 1j * k * fk
    np.testing.assert_allclose(result, expected)


def test_dfdk_shape_preserved():
    """Output shape should match fk shape."""
    k = np.linspace(0, 5, 10)
    fk = np.ones_like(k, dtype=complex)
    result = dfdk(k, fk)
    assert result.shape == fk.shape


def test_dfdk2_no_k2():
    """Test second derivative with single k."""
    k = np.array([1.0, 2.0])
    fk = np.array([1.0 + 0j, 2.0 + 0j])
    result = dfdk2(k, fk)
    expected = -(k**2) * fk
    np.testing.assert_allclose(result, expected)


def test_dfdk2_with_k2():
    """Test mixed derivative with k1 and k2."""
    k1 = np.array([1.0, 2.0])
    k2 = np.array([3.0, 4.0])
    fk = np.array([1.0 + 0j, 2.0 + 0j])
    result = dfdk2(k1, fk, k2)
    expected = -k1 * k2 * fk
    np.testing.assert_allclose(result, expected)


def test_dfdk2_shape_preserved():
    """Output shape should match fk shape."""
    k1 = np.linspace(0, 5, 10)
    fk = np.ones_like(k1, dtype=complex)
    result = dfdk2(k1, fk)
    assert result.shape == fk.shape


def test_dfdk_and_dfdk2_consistency():
    """Second derivative should match applying dfdk twice (up to 1j factors)."""
    k = np.array([1.0, 2.0, 3.0])
    fk = np.array([1.0 + 0j, -1.0 + 0j, 2.0 + 0j])

    first_derivative = dfdk(k, fk)
    second_by_dfdk = dfdk(k, first_derivative)
    second_by_dfdk2 = dfdk2(k, fk)

    np.testing.assert_allclose(second_by_dfdk, second_by_dfdk2)
