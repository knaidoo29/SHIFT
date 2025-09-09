import numpy as np
import pytest

import shift


def test_mpi_kgrid2D():
    
    mpi = shift.mpiutils.MPI()

    boxsize = 100
    ngrid = 16

    kx2D, ky2D = shift.cart.mpi_kgrid2D(boxsize, ngrid, mpi)
    kx2D, ky2D = shift.cart.mpi_kgrid2D_dct(boxsize, ngrid, mpi)
    kx2D, ky2D = shift.cart.mpi_kgrid2D_dst(boxsize, ngrid, mpi)
    
    boxsize = [100, 50]
    ngrid = [16, 8]

    kx2D, ky2D = shift.cart.mpi_kgrid2D(boxsize, ngrid, mpi)
    kx2D, ky2D = shift.cart.mpi_kgrid2D_dct(boxsize, ngrid, mpi)
    kx2D, ky2D = shift.cart.mpi_kgrid2D_dst(boxsize, ngrid, mpi)


def test_mpi_kgrid3D():
    
    mpi = shift.mpiutils.MPI()

    boxsize = 100
    ngrid = 16

    kx2D, ky2D, kz3D = shift.cart.mpi_kgrid3D(boxsize, ngrid, mpi)
    kx2D, ky2D, kz3D = shift.cart.mpi_kgrid3D_dct(boxsize, ngrid, mpi)
    kx2D, ky2D, kz3D = shift.cart.mpi_kgrid3D_dst(boxsize, ngrid, mpi)
    
    boxsize = [100, 50, 25]
    ngrid = [16, 8, 4]

    kx2D, ky2D, kz3D = shift.cart.mpi_kgrid3D(boxsize, ngrid, mpi)
    kx2D, ky2D, kz3D = shift.cart.mpi_kgrid3D_dct(boxsize, ngrid, mpi)
    kx2D, ky2D, kz3D = shift.cart.mpi_kgrid3D_dst(boxsize, ngrid, mpi)

