import numpy as np
import pytest

from shift.cart import (
    get_kf,
    kgrid1D, kgrid2D, kgrid3D,
    kgrid1D_dct, kgrid2D_dct, kgrid3D_dct,
    kgrid1D_dst, kgrid2D_dst, kgrid3D_dst,
)

# -----------------
# Fourier Grids
# -----------------

def test_kgrid1D_symmetric():
    boxsize, ngrid = 10.0, 4
    k = kgrid1D(boxsize, ngrid)
    kf = get_kf(boxsize)
    expected = np.array([0, 1, -2, -1]) * kf
    np.testing.assert_allclose(k, expected)


def test_kgrid2D_shape_and_consistency():
    boxsize, ngrid = 8.0, 4
    kx2D, ky2D = kgrid2D(boxsize, ngrid)
    assert kx2D.shape == (ngrid, ngrid)
    assert ky2D.shape == (ngrid, ngrid)
    # First column of kx2D equals kgrid1D
    np.testing.assert_allclose(kx2D[:, 0], kgrid1D(boxsize, ngrid))


def test_kgrid2D_shape_and_consistency_v2():
    boxsize, ngrid = [8.0, 4.], [4, 8]
    kx2D, ky2D = kgrid2D(boxsize, ngrid)
    assert kx2D.shape == (ngrid[0], ngrid[1])
    assert ky2D.shape == (ngrid[0], ngrid[1])
    # First column of kx2D equals kgrid1D
    np.testing.assert_allclose(kx2D[:, 0], kgrid1D(boxsize[0], ngrid[0]))


def test_kgrid3D_shape_and_consistency():
    boxsize, ngrid = 6.0, 3
    kx3D, ky3D, kz3D = kgrid3D(boxsize, ngrid)
    assert kx3D.shape == (ngrid, ngrid, ngrid)
    assert ky3D.shape == (ngrid, ngrid, ngrid)
    assert kz3D.shape == (ngrid, ngrid, ngrid)
    # First line along x equals kgrid1D
    np.testing.assert_allclose(kx3D[:, 0, 0], kgrid1D(boxsize, ngrid))


def test_kgrid3D_shape_and_consistency_v2():
    boxsize, ngrid = [6.0, 12.0, 24.0], [3, 6, 9]
    kx3D, ky3D, kz3D = kgrid3D(boxsize, ngrid)
    assert kx3D.shape == (ngrid[0], ngrid[1], ngrid[2])
    assert ky3D.shape == (ngrid[0], ngrid[1], ngrid[2])
    assert kz3D.shape == (ngrid[0], ngrid[1], ngrid[2])
    # First line along x equals kgrid1D
    np.testing.assert_allclose(kx3D[:, 0, 0], kgrid1D(boxsize[0], ngrid[0]))


# -----------------
# DCT Grids
# -----------------

def test_kgrid1D_dct_starts_at_zero():
    boxsize, ngrid = 10.0, 5
    k = kgrid1D_dct(boxsize, ngrid)
    kf = get_kf(boxsize)
    expected = np.arange(ngrid) * kf/2
    np.testing.assert_allclose(k, expected)


def test_kgrid2D_dct_shape_and_consistency():
    boxsize, ngrid = 12.0, 3
    kx2D, ky2D = kgrid2D_dct(boxsize, ngrid)
    assert kx2D.shape == (ngrid, ngrid)
    assert ky2D.shape == (ngrid, ngrid)
    np.testing.assert_allclose(kx2D[:, 0], kgrid1D_dct(boxsize, ngrid))


def test_kgrid2D_dct_shape_and_consistency_v2():
    boxsize, ngrid = [12.0, 24.0], [3, 6]
    kx2D, ky2D = kgrid2D_dct(boxsize, ngrid)
    assert kx2D.shape == (ngrid[0], ngrid[1])
    assert ky2D.shape == (ngrid[0], ngrid[1])
    np.testing.assert_allclose(kx2D[:, 0], kgrid1D_dct(boxsize[0], ngrid[0]))


def test_kgrid3D_dct_shape_and_consistency():
    boxsize, ngrid = 9.0, 2
    kx3D, ky3D, kz3D = kgrid3D_dct(boxsize, ngrid)
    assert kx3D.shape == (ngrid, ngrid, ngrid)
    assert ky3D.shape == (ngrid, ngrid, ngrid)
    assert kz3D.shape == (ngrid, ngrid, ngrid)
    np.testing.assert_allclose(kx3D[:, 0, 0], kgrid1D_dct(boxsize, ngrid))


def test_kgrid3D_dct_shape_and_consistency_v2():
    boxsize, ngrid = [9.0, 18.0, 36.], [2, 4, 8]
    kx3D, ky3D, kz3D = kgrid3D_dct(boxsize, ngrid)
    assert kx3D.shape == (ngrid[0], ngrid[1], ngrid[2])
    assert ky3D.shape == (ngrid[0], ngrid[1], ngrid[2])
    assert kz3D.shape == (ngrid[0], ngrid[1], ngrid[2])
    np.testing.assert_allclose(kx3D[:, 0, 0], kgrid1D_dct(boxsize[0], ngrid[0]))


# -----------------
# DST Grids
# -----------------

def test_kgrid1D_dst_starts_at_kf_over_2():
    boxsize, ngrid = 8.0, 4
    k = kgrid1D_dst(boxsize, ngrid)
    kf = get_kf(boxsize)
    expected = np.arange(1, ngrid+1) * kf/2
    np.testing.assert_allclose(k, expected)


def test_kgrid2D_dst_shape_and_consistency():
    boxsize, ngrid = 6.0, 3
    kx2D, ky2D = kgrid2D_dst(boxsize, ngrid)
    assert kx2D.shape == (ngrid, ngrid)
    assert ky2D.shape == (ngrid, ngrid)
    np.testing.assert_allclose(kx2D[:, 0], kgrid1D_dst(boxsize, ngrid))


def test_kgrid2D_dst_shape_and_consistency_v2():
    boxsize, ngrid = [6.0, 12.0], [3, 6]
    kx2D, ky2D = kgrid2D_dst(boxsize, ngrid)
    assert kx2D.shape == (ngrid[0], ngrid[1])
    assert ky2D.shape == (ngrid[0], ngrid[1])
    np.testing.assert_allclose(kx2D[:, 0], kgrid1D_dst(boxsize[0], ngrid[0]))


def test_kgrid3D_dst_shape_and_consistency():
    boxsize, ngrid = 5.0, 2
    kx3D, ky3D, kz3D = kgrid3D_dst(boxsize, ngrid)
    assert kx3D.shape == (ngrid, ngrid, ngrid)
    np.testing.assert_allclose(ky3D[0, :, 0], kgrid1D_dst(boxsize, ngrid))


def test_kgrid3D_dst_shape_and_consistency_v2():
    boxsize, ngrid = [5.0, 10.0, 20.0], [2, 4, 8]
    kx3D, ky3D, kz3D = kgrid3D_dst(boxsize, ngrid)
    assert kx3D.shape == (ngrid[0], ngrid[1], ngrid[2])
    assert ky3D.shape == (ngrid[0], ngrid[1], ngrid[2])
    assert kz3D.shape == (ngrid[0], ngrid[1], ngrid[2])
    np.testing.assert_allclose(kx3D[:, 0, 0], kgrid1D_dst(boxsize[0], ngrid[0]))