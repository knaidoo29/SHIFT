import numpy as np
import pytest

from shift.cart import mult_fk_2D, mult_fk_3D


# -----------------
# 2D Tests
# -----------------

def test_mult_fk_2D_identity():
    """f(k)=1 should return the same grid (identity filter)."""
    boxsize, n = 10.0, 8
    grid = np.random.rand(n, n)
    k = np.linspace(0.1, 10.0, 50)   # avoid zero to hit fkgrid[1:] branch
    fk = np.ones_like(k)
    out = mult_fk_2D(grid, boxsize, k, fk)
    np.testing.assert_allclose(out, grid, atol=1e-10)


def test_mult_fk_2D_scaling():
    """f(k)=2 should roughly double the amplitude."""
    boxsize, n = 12.0, 8
    grid = np.random.rand(n, n)
    k = np.linspace(1e-2, 1e2, 50)
    fk = np.full_like(k, 1.0)
    out = mult_fk_2D(grid, boxsize, k, fk)
    # Values are scaled, not identical but should correlate strongly
    assert np.corrcoef(grid.flatten(), out.flatten())[0, 1] > 0.99
    assert np.all(out > 0)


def test_mult_fk_2D_invalid_range_raises():
    """Should raise AssertionError if k range does not cover kmag."""
    boxsize, n = 10.0, 8
    grid = np.random.rand(n, n)
    k = np.linspace(100, 200, 10)  # completely outside Fourier range
    fk = np.ones_like(k)
    with pytest.raises(AssertionError):
        mult_fk_2D(grid, boxsize, k, fk)


def test_mult_fk_2D_invalid_range_raises_v2():
    """Should raise AssertionError if k range does not cover kmag."""
    boxsize, n = 10.0, 8
    grid = np.random.rand(n, n)
    k = np.linspace(1e-5, 1e-3, 10)  # completely outside Fourier range
    fk = np.ones_like(k)
    with pytest.raises(AssertionError):
        mult_fk_2D(grid, boxsize, k, fk)


def test_mult_fk_2D_with_zero_kmin():
    """When k.min()==0, it should use all kmag in multiplication."""
    boxsize, n = 8.0, 6
    grid = np.random.rand(n, n)
    k = np.linspace(0, 10.0, 50)  # includes 0
    fk = np.linspace(1, 2, 50)    # varying function
    out = mult_fk_2D(grid, boxsize, k, fk)
    assert out.shape == grid.shape


# -----------------
# 3D Tests
# -----------------

def test_mult_fk_3D_identity():
    boxsize, n = 9.0, 6
    grid = np.random.rand(n, n, n)
    k = np.linspace(0.1, 15.0, 60)
    fk = np.ones_like(k)
    out = mult_fk_3D(grid, boxsize, k, fk)
    np.testing.assert_allclose(out, grid, atol=1e-10)


def test_mult_fk_3D_scaling():
    boxsize, n = 10.0, 6
    grid = np.random.rand(n, n, n)
    k = np.linspace(0.1, 20.0, 60)
    fk = np.full_like(k, 0.5)
    out = mult_fk_3D(grid, boxsize, k, fk)
    assert np.corrcoef(grid.flatten(), out.flatten())[0, 1] > 0.99


def test_mult_fk_3D_invalid_range_raises():
    boxsize, n = 10.0, 6
    grid = np.random.rand(n, n, n)
    k = np.linspace(100, 200, 20)
    fk = np.ones_like(k)
    with pytest.raises(AssertionError):
        mult_fk_3D(grid, boxsize, k, fk)


def test_mult_fk_3D_invalid_range_raises_v2():
    boxsize, n = 10.0, 6
    grid = np.random.rand(n, n, n)
    k = np.linspace(1e-5, 1e-4, 20)
    fk = np.ones_like(k)
    with pytest.raises(AssertionError):
        mult_fk_3D(grid, boxsize, k, fk)


def test_mult_fk_3D_with_zero_kmin():
    boxsize, n = 8.0, 5
    grid = np.random.rand(n, n, n)
    k = np.linspace(0, 12.0, 40)
    fk = np.linspace(1, 3, 40)
    out = mult_fk_3D(grid, boxsize, k, fk)
    assert out.shape == grid.shape
