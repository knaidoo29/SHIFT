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