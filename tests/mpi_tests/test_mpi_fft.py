import numpy as np
import pytest

import shift


def test_mpi_fft2D_roundtrip():
    
    mpi = shift.mpiutils.MPI()

    boxsize = 100
    ngrid = 16

    x3D, y3D = shift.cart.mpi_grid2D(boxsize, ngrid, mpi)
    f = np.random.normal(size=np.shape(x3D))

    fk = shift.cart.mpi_fft2D(f, boxsize, ngrid, mpi)
    f2 = shift.cart.mpi_ifft2D(fk, boxsize, ngrid, mpi)

    np.testing.assert_allclose(f, f2, rtol=1e-12, atol=1e-12)

    boxsize = [100, 50]
    ngrid = [16, 8]

    x3D, y3D = shift.cart.mpi_grid2D(boxsize, ngrid, mpi)
    f = np.random.normal(size=np.shape(x3D))

    fk = shift.cart.mpi_fft2D(f, boxsize, ngrid, mpi)
    f2 = shift.cart.mpi_ifft2D(fk, boxsize, ngrid, mpi)

    np.testing.assert_allclose(f, f2, rtol=1e-12, atol=1e-12)



def test_mpi_dct2D_roundtrip():
    
    mpi = shift.mpiutils.MPI()

    boxsize = 100
    ngrid = 16

    x3D, y3D = shift.cart.mpi_grid2D(boxsize, ngrid, mpi)
    f = np.random.normal(size=np.shape(x3D))

    fk = shift.cart.mpi_dct2D(f, boxsize, ngrid, mpi)
    f2 = shift.cart.mpi_idct2D(fk, boxsize, ngrid, mpi)

    np.testing.assert_allclose(f, f2, rtol=1e-12, atol=1e-12)

    boxsize = [100, 50]
    ngrid = [16, 8]

    x3D, y3D = shift.cart.mpi_grid2D(boxsize, ngrid, mpi)
    f = np.random.normal(size=np.shape(x3D))

    fk = shift.cart.mpi_dct2D(f, boxsize, ngrid, mpi)
    f2 = shift.cart.mpi_idct2D(fk, boxsize, ngrid, mpi)

    np.testing.assert_allclose(f, f2, rtol=1e-12, atol=1e-12)



def test_mpi_dst2D_roundtrip():
    
    mpi = shift.mpiutils.MPI()

    boxsize = 100
    ngrid = 16

    x3D, y3D = shift.cart.mpi_grid2D(boxsize, ngrid, mpi)
    f = np.random.normal(size=np.shape(x3D))

    fk = shift.cart.mpi_dst2D(f, boxsize, ngrid, mpi)
    f2 = shift.cart.mpi_idst2D(fk, boxsize, ngrid, mpi)

    np.testing.assert_allclose(f, f2, rtol=1e-12, atol=1e-12)

    boxsize = [100, 50]
    ngrid = [16, 8]

    x3D, y3D = shift.cart.mpi_grid2D(boxsize, ngrid, mpi)
    f = np.random.normal(size=np.shape(x3D))

    fk = shift.cart.mpi_dst2D(f, boxsize, ngrid, mpi)
    f2 = shift.cart.mpi_idst2D(fk, boxsize, ngrid, mpi)

    np.testing.assert_allclose(f, f2, rtol=1e-12, atol=1e-12)


def test_mpi_fft3D_roundtrip():
    
    mpi = shift.mpiutils.MPI()

    boxsize = 100
    ngrid = 16

    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, mpi)
    f = np.random.normal(size=np.shape(x3D))

    fk = shift.cart.mpi_fft3D(f, boxsize, ngrid, mpi)
    f2 = shift.cart.mpi_ifft3D(fk, boxsize, ngrid, mpi)

    np.testing.assert_allclose(f, f2, rtol=1e-12, atol=1e-12)

    boxsize = [100, 50, 25]
    ngrid = [16, 8, 4]
    
    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, mpi)

    f = np.random.normal(size=np.shape(x3D))

    fk = shift.cart.mpi_fft3D(f, boxsize, ngrid, mpi)
    f2 = shift.cart.mpi_ifft3D(fk, boxsize, ngrid, mpi)

    np.testing.assert_allclose(f, f2, rtol=1e-12, atol=1e-12)


def test_mpi_dct3D_roundtrip():
    
    mpi = shift.mpiutils.MPI()

    boxsize = 100
    ngrid = 16

    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, mpi)
    f = np.random.normal(size=np.shape(x3D))

    fk = shift.cart.mpi_dct3D(f, boxsize, ngrid, mpi)
    f2 = shift.cart.mpi_idct3D(fk, boxsize, ngrid, mpi)

    np.testing.assert_allclose(f, f2, rtol=1e-12, atol=1e-12)

    boxsize = [100, 50, 25]
    ngrid = [16, 8, 4]
    
    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, mpi)

    f = np.random.normal(size=np.shape(x3D))

    fk = shift.cart.mpi_dct3D(f, boxsize, ngrid, mpi)
    f2 = shift.cart.mpi_idct3D(fk, boxsize, ngrid, mpi)

    np.testing.assert_allclose(f, f2, rtol=1e-12, atol=1e-12)



def test_mpi_dst3D_roundtrip():
    
    mpi = shift.mpiutils.MPI()

    boxsize = 100
    ngrid = 16

    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, mpi)
    f = np.random.normal(size=np.shape(x3D))

    fk = shift.cart.mpi_dst3D(f, boxsize, ngrid, mpi)
    f2 = shift.cart.mpi_idst3D(fk, boxsize, ngrid, mpi)

    np.testing.assert_allclose(f, f2, rtol=1e-12, atol=1e-12)

    boxsize = [100, 50, 25]
    ngrid = [16, 8, 4]

    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, mpi)

    f = np.random.normal(size=np.shape(x3D))

    fk = shift.cart.mpi_dst3D(f, boxsize, ngrid, mpi)
    f2 = shift.cart.mpi_idst3D(fk, boxsize, ngrid, mpi)

    np.testing.assert_allclose(f, f2, rtol=1e-12, atol=1e-12)

import scipy.fft as scfft
from shift.cart.mpi_fft import (
    redistribute_forward_2D,
    redistribute_forward_3D,
    redistribute_backward_2D,
    redistribute_backward_3D,
)


def _redistribution_grid(size, ndim, grid_mode):
    """Choose grids exercising both equal and unequal MPI slab widths."""
    if grid_mode == "equal_scalar":
        ngrid = 4 * size
        shape = (ngrid,) * ndim
    elif grid_mode == "equal_rectangular":
        shape = tuple(k * size for k in (4, 3, 2)[:ndim])
        ngrid = list(shape)
    elif grid_mode == "unequal_rectangular":
        shape = tuple(k * size + 1 for k in (4, 3, 2)[:ndim])
        ngrid = list(shape)
    else:
        raise ValueError(grid_mode)
    return ngrid, shape


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("grid_mode", ["equal_scalar", "equal_rectangular", "unequal_rectangular"])
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_redistribute_matches_global_array(ndim, grid_mode, dtype):
    """Check x-to-y and y-to-x redistributions against known global slices."""
    mpi = shift.mpiutils.MPI()
    ngrid, shape = _redistribution_grid(mpi.size, ndim, grid_mode)

    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        global_data = (values + 1j * (values + 7)).astype(dtype)
    else:
        global_data = values.astype(dtype)

    xstart, xstop = mpi.split(shape[0])
    ystart, ystop = mpi.split(shape[1])

    local_x = np.ascontiguousarray(global_data[xstart[mpi.rank]:xstop[mpi.rank]])
    expected_y = np.ascontiguousarray(global_data[:, ystart[mpi.rank]:ystop[mpi.rank]])

    if ndim == 2:
        got_y = redistribute_forward_2D(local_x, ngrid, mpi)
        got_x = redistribute_backward_2D(got_y, ngrid, mpi)
    else:
        got_y = redistribute_forward_3D(local_x, ngrid, mpi)
        got_x = redistribute_backward_3D(got_y, ngrid, mpi)

    # Evaluate *after* all MPI calls, then share results with all ranks. This
    # prevents rank-dependent assertion failures from desynchronising pytest.
    local_ok = (
        got_y.dtype == global_data.dtype
        and got_x.dtype == global_data.dtype
        and np.array_equal(got_y, expected_y)
        and np.array_equal(got_x, local_x)
    )
    failures = mpi.comm.allgather(None if local_ok else mpi.rank)
    assert all(rank is None for rank in failures), (
        f"Redistribution mismatch on ranks {[rank for rank in failures if rank is not None]} "
        f"(ndim={ndim}, mode={grid_mode}, dtype={np.dtype(dtype)})"
    )


@pytest.mark.parametrize(
    "function,ndim",
    [
        (redistribute_forward_2D, 2),
        (redistribute_backward_2D, 2),
        (redistribute_forward_3D, 3),
        (redistribute_backward_3D, 3),
    ],
)
@pytest.mark.parametrize("failure", ["wrong_grid_length", "wrong_local_shape"])
def test_redistribute_rejects_invalid_inputs(function, ndim, failure):
    """Exercise dimensionality and local-shape validation on every rank."""
    mpi = shift.mpiutils.MPI()
    _, shape = _redistribution_grid(mpi.size, ndim, "equal_rectangular")
    local = np.zeros((1,) * ndim, dtype=np.float64)

    if failure == "wrong_grid_length":
        bad_grid = list(shape[:-1])
        with pytest.raises(ValueError, match="Length of grid dimensions"):
            function(local, bad_grid, mpi)
    else:
        with pytest.raises(ValueError, match="Expected local shape"):
            function(local, list(shape), mpi)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("kind", ["fft", "dct", "dst"])
@pytest.mark.parametrize("grid_mode", ["equal_rectangular", "unequal_rectangular"])
def test_forward_transform_matches_serial_scipy(ndim, kind, grid_mode):
    """Check distributed transforms against an independent full-grid transform."""
    mpi = shift.mpiutils.MPI()
    ngrid, shape = _redistribution_grid(mpi.size, ndim, grid_mode)
    boxsize = [120.0, 90.0, 60.0][:ndim]

    global_data = np.random.default_rng(1234).normal(size=shape)
    xstart, xstop = mpi.split(shape[0])
    ystart, ystop = mpi.split(shape[1])
    local_x = np.ascontiguousarray(global_data[xstart[mpi.rank]:xstop[mpi.rank]])

    funcs = {
        (2, "fft"): shift.cart.mpi_fft2D,
        (2, "dct"): shift.cart.mpi_dct2D,
        (2, "dst"): shift.cart.mpi_dst2D,
        (3, "fft"): shift.cart.mpi_fft3D,
        (3, "dct"): shift.cart.mpi_dct3D,
        (3, "dst"): shift.cart.mpi_dst3D,
    }
    got = funcs[(ndim, kind)](local_x, boxsize, ngrid, mpi)

    if kind == "fft":
        expected_global = scfft.fftn(global_data)
    elif kind == "dct":
        expected_global = scfft.dctn(global_data, type=2)
    else:
        expected_global = scfft.dstn(global_data, type=2)

    # The SHIFT transforms multiply each axis by dx / sqrt(2*pi).
    spacing = np.asarray(boxsize) / np.asarray(shape)
    expected_global *= float(np.prod(spacing) / (2.0 * np.pi) ** (ndim / 2))
    expected_y = expected_global[:, ystart[mpi.rank]:ystop[mpi.rank]]

    local_ok = np.allclose(got, expected_y, rtol=1e-11, atol=1e-10)
    failures = mpi.comm.allgather(None if local_ok else mpi.rank)
    assert all(rank is None for rank in failures), (
        f"Forward {kind} mismatch on ranks {[rank for rank in failures if rank is not None]} "
        f"(ndim={ndim}, mode={grid_mode})"
    )
