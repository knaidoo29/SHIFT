import numpy as np
import pytest

from shift.cart import (
    fft1D, ifft1D, fft2D, ifft2D, fft3D, ifft3D,
    dct1D, idct1D, dct2D, idct2D, dct3D, idct3D,
    dst1D, idst1D, dst2D, idst2D, dst3D, idst3D,
)


# --------------------
# 1D FFT Tests
# --------------------

def test_fft1D_ifft1D_roundtrip():
    """Forward and backward FFT should recover the original array (1D)."""
    x = np.linspace(0, 1, 16)
    boxsize = 1.0
    f = fft1D(x, boxsize)
    x_recovered = ifft1D(f, boxsize)
    np.testing.assert_allclose(x, x_recovered, rtol=1e-12, atol=1e-12)


def test_fft1D_shape_preserved():
    x = np.random.rand(32)
    boxsize = 2.0
    f = fft1D(x, boxsize)
    assert f.shape == x.shape


# --------------------
# 2D FFT Tests
# --------------------

def test_fft2D_ifft2D_roundtrip_scalar_boxsize():
    """Roundtrip with scalar boxsize in 2D."""
    x = np.random.rand(8, 8)
    boxsize = 2.0
    f = fft2D(x, boxsize)
    x_recovered = ifft2D(f, boxsize)
    np.testing.assert_allclose(x, x_recovered, rtol=1e-12, atol=1e-12)


def test_fft2D_ifft2D_roundtrip_nonvector_boxsize():
    """Roundtrip with per-axis boxsize in 2D."""
    x = np.random.rand(6, 10)
    boxsize = 3.0
    f = fft2D(x, boxsize)
    x_recovered = ifft2D(f, boxsize)
    np.testing.assert_allclose(x, x_recovered, rtol=1e-12, atol=1e-12)

def test_fft2D_ifft2D_roundtrip_vector_boxsize():
    """Roundtrip with per-axis boxsize in 2D."""
    x = np.random.rand(6, 10)
    boxsize = [2.0, 3.0]
    f = fft2D(x, boxsize)
    x_recovered = ifft2D(f, boxsize)
    np.testing.assert_allclose(x, x_recovered, rtol=1e-12, atol=1e-12)


def test_fft2D_shape_preserved():
    x = np.random.rand(5, 7)
    f = fft2D(x, [1.0, 1.0])
    assert f.shape == x.shape


# --------------------
# 3D FFT Tests
# --------------------

def test_fft3D_ifft3D_roundtrip_scalar_boxsize():
    x = np.random.rand(4, 4, 4)
    boxsize = 1.0
    f = fft3D(x, boxsize)
    x_recovered = ifft3D(f, boxsize)
    np.testing.assert_allclose(x, x_recovered, rtol=1e-12, atol=1e-12)


def test_fft3D_ifft3D_roundtrip_nonvector_boxsize():
    x = np.random.rand(3, 4, 5)
    boxsize = 1.0
    f = fft3D(x, boxsize)
    x_recovered = ifft3D(f, boxsize)
    np.testing.assert_allclose(x, x_recovered, rtol=1e-12, atol=1e-12)


def test_fft3D_ifft3D_roundtrip_vector_boxsize():
    x = np.random.rand(3, 4, 5)
    boxsize = [1.0, 2.0, 3.0]
    f = fft3D(x, boxsize)
    x_recovered = ifft3D(f, boxsize)
    np.testing.assert_allclose(x, x_recovered, rtol=1e-12, atol=1e-12)


def test_fft3D_shape_preserved():
    x = np.random.rand(6, 7, 8)
    f = fft3D(x, [1.0, 2.0, 3.0])
    assert f.shape == x.shape


# --------------------
# DCT Tests
# --------------------

def test_dct1D_idct1D_roundtrip():
    x = np.random.rand(16)
    boxsize = 2.0
    f = dct1D(x, boxsize, type=2)
    x_rec = idct1D(f, boxsize, type=2)
    np.testing.assert_allclose(x, x_rec, rtol=1e-12, atol=1e-12)


def test_dct2D_idct2D_roundtrip():
    x = np.random.rand(6, 8)
    boxsize = 3.0
    f = dct2D(x, boxsize, type=2)
    x_rec = idct2D(f, boxsize, type=2)
    np.testing.assert_allclose(x, x_rec, rtol=1e-12, atol=1e-12)


def test_dct2D_idct2D_vector_boxsize_roundtrip():
    x = np.random.rand(6, 8)
    boxsize = [2.0, 3.0]
    f = dct2D(x, boxsize, type=2)
    x_rec = idct2D(f, boxsize, type=2)
    np.testing.assert_allclose(x, x_rec, rtol=1e-12, atol=1e-12)


def test_dct3D_idct3D_roundtrip():
    x = np.random.rand(4, 5, 6)
    boxsize = 1.0
    f = dct3D(x, boxsize, type=2)
    x_rec = idct3D(f, boxsize, type=2)
    np.testing.assert_allclose(x, x_rec, rtol=1e-12, atol=1e-12)


def test_dct3D_idct3D_vector_boxsize_roundtrip():
    x = np.random.rand(4, 5, 6)
    boxsize = [1.0, 2.0, 3.0]
    f = dct3D(x, boxsize, type=2)
    x_rec = idct3D(f, boxsize, type=2)
    np.testing.assert_allclose(x, x_rec, rtol=1e-12, atol=1e-12)


def test_dct1D_constant():
    """DCT of constant should produce only first coefficient nonzero."""
    x = np.ones(8)
    f = dct1D(x, boxsize=1.0, type=2)
    assert np.isclose(f[0], f[0].real)
    assert np.allclose(f[1:], 0, atol=1e-12)


# --------------------
# DST Tests
# --------------------

def test_dst1D_idst1D_roundtrip():
    x = np.random.rand(16)
    boxsize = 2.0
    f = dst1D(x, boxsize, type=2)
    x_rec = idst1D(f, boxsize, type=2)
    np.testing.assert_allclose(x, x_rec, rtol=1e-12, atol=1e-12)


def test_dst2D_idst2D_roundtrip():
    x = np.random.rand(5, 7)
    boxsize = 2.0
    f = dst2D(x, boxsize, type=2)
    x_rec = idst2D(f, boxsize, type=2)
    np.testing.assert_allclose(x, x_rec, rtol=1e-12, atol=1e-12)


def test_dst2D_idst2D_vector_boxsize_roundtrip():
    x = np.random.rand(5, 7)
    boxsize = [2.0, 3.0]
    f = dst2D(x, boxsize, type=2)
    x_rec = idst2D(f, boxsize, type=2)
    np.testing.assert_allclose(x, x_rec, rtol=1e-12, atol=1e-12)


def test_dst3D_idst3D_roundtrip():
    x = np.random.rand(3, 4, 5)
    boxsize = 1.0
    f = dst3D(x, boxsize, type=2)
    x_rec = idst3D(f, boxsize, type=2)
    np.testing.assert_allclose(x, x_rec, rtol=1e-12, atol=1e-12)

def test_dst3D_idst3D_vector_boxsize_roundtrip():
    x = np.random.rand(3, 4, 5)
    boxsize = [1.0, 2.0, 3.0]
    f = dst3D(x, boxsize, type=2)
    x_rec = idst3D(f, boxsize, type=2)
    np.testing.assert_allclose(x, x_rec, rtol=1e-12, atol=1e-12)


def test_shapes_preserved():
    """Ensure forward/backward transforms keep the same shape."""
    for func_fwd, func_inv, shape, boxsize in [
        (fft1D, ifft1D, (10,), 1.0),
        (fft2D, ifft2D, (8, 12), [1.0, 2.0]),
        (fft3D, ifft3D, (4, 5, 6), [1.0, 1.0, 1.0]),
        (dct1D, idct1D, (10,), 1.0),
        (dct2D, idct2D, (8, 12), [1.0, 2.0]),
        (dct3D, idct3D, (4, 5, 6), [1.0, 1.0, 1.0]),
        (dst1D, idst1D, (10,), 1.0),
        (dst2D, idst2D, (6, 6), [2.0, 2.0]),
        (dst3D, idst3D, (3, 4, 5), [1.0, 2.0, 3.0]),
    ]:
        x = np.random.rand(*shape)
        f = func_fwd(x, boxsize)
        x_rec = func_inv(f, boxsize)
        assert x.shape == f.shape == x_rec.shape