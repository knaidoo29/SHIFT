import numpy as np
import pytest

from shift.cart import get_pofk_2D, get_pofk_3D


# -----------------
# Power spectrum 2D
# -----------------

def test_get_pofk_2D_uniform_field():
    boxsize, ngrid = 10.0, 8
    dgrid = np.ones((ngrid, ngrid))
    k, keff, pk = get_pofk_2D(dgrid, boxsize, ngrid)
    assert len(k) == len(keff) == len(pk)
    # uniform field has zero fluctuations
    assert np.allclose(pk[1:], 0.0, atol=1e-12)


def test_get_pofk_2D_delta_field():
    boxsize, ngrid = 10.0, 8
    dgrid = np.zeros((ngrid, ngrid))
    dgrid[0, 0] = 1.0
    k, keff, pk = get_pofk_2D(dgrid, boxsize, ngrid)
    assert np.all(pk >= 0)
    assert pk.max() > 0


def test_get_pofk_2D_with_kmin_kmax():
    boxsize, ngrid = 10.0, 8
    dgrid = np.random.rand(ngrid, ngrid)
    kmin = 2 * np.pi / boxsize
    kmax = 3 * kmin
    k, keff, pk = get_pofk_2D(dgrid, boxsize, ngrid, kmin=kmin, kmax=kmax)
    assert k.min() >= kmin
    assert k.max() <= kmax


# -----------------
# Power spectrum 3D
# -----------------

def test_get_pofk_3D_uniform_field():
    boxsize, ngrid = 12.0, 6
    dgrid = np.ones((ngrid, ngrid, ngrid))
    k, keff, pk = get_pofk_3D(dgrid, boxsize, ngrid)
    assert len(k) == len(keff) == len(pk)
    assert np.allclose(pk[1:], 0.0, atol=1e-12)


def test_get_pofk_3D_delta_field():
    boxsize, ngrid = 12.0, 6
    dgrid = np.zeros((ngrid, ngrid, ngrid))
    dgrid[0, 0, 0] = 1.0
    k, keff, pk = get_pofk_3D(dgrid, boxsize, ngrid)
    assert np.all(pk >= 0)
    assert pk.max() > 0


def test_get_pofk_3D_with_kmin_kmax():
    boxsize, ngrid = 12.0, 6
    dgrid = np.random.rand(ngrid, ngrid, ngrid)
    kmin = 2 * np.pi / boxsize
    kmax = 4 * kmin
    k, keff, pk = get_pofk_3D(dgrid, boxsize, ngrid, kmin=kmin, kmax=kmax)
    assert k.min() >= kmin
    assert k.max() <= kmax
