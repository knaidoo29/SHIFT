import pytest
import numpy as np
from shift import mpiutils


def test_get_mpi_loop_size_exact_division():
    # 100 iterations, 10 ranks -> exactly 10 per rank
    assert mpiutils.get_MPI_loop_size(100, 10) == 10


def test_get_mpi_loop_size_not_divisible():
    # 101 iterations, 10 ranks -> should round up to 11
    assert mpiutils.get_MPI_loop_size(101, 10) == 11


def test_get_mpi_loop_size_smaller_loop_than_ranks():
    # 5 iterations, 10 ranks -> each rank needs at most 1
    assert mpiutils.get_MPI_loop_size(5, 10) == 1


def test_get_mpi_loop_size_one_rank():
    # Only one rank -> entire loop goes to it
    assert mpiutils.get_MPI_loop_size(50, 1) == 50


def test_get_mpi_loop_size_large_loop_large_ranks():
    result = mpiutils.get_MPI_loop_size(1_000_000, 256)
    expected = int(np.ceil(1_000_000 / 256))
    assert result == expected
