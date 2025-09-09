import numpy as np
import pytest

import shift


def test_mpi_grid2D():
    
    mpi = shift.mpiutils.MPI()

    boxsize = 100
    ngrid = 16

    x2D, y2D = shift.cart.mpi_grid2D(boxsize, ngrid, mpi)
    x2D, y2D = shift.cart.mpi_grid2D(boxsize, ngrid, mpi, origin=1.)
    x2D, y2D = shift.cart.mpi_grid2D(boxsize, ngrid, mpi, origin=[1., 2.])
    
    boxsize = [100, 50]
    ngrid = [16, 8]

    x2D, y2D = shift.cart.mpi_grid2D(boxsize, ngrid, mpi)
    x2D, y2D = shift.cart.mpi_grid2D(boxsize, ngrid, mpi, origin=1.)
    x2D, y2D = shift.cart.mpi_grid2D(boxsize, ngrid, mpi, origin=[1., 2.])


def test_mpi_grid3D():
    
    mpi = shift.mpiutils.MPI()

    boxsize = 100
    ngrid = 16

    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, mpi)
    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, mpi, origin=1.)
    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, mpi, origin=[1., 2., 4.])
    
    boxsize = [100, 50, 25]
    ngrid = [16, 8, 4]

    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, mpi)
    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, mpi, origin=1.)
    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, mpi, origin=[1., 2., 4.])
    
    