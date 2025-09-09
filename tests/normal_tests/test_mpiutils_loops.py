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


def test_mpi_ind2ind_valid_mapping():
    # 10 loop iterations, 4 ranks
    # MPI_ind = 2, rank = 1 -> global index = 2*4+1 = 9
    assert mpiutils.MPI_ind2ind(2, 1, 4, 10) == 9


def test_mpi_ind2ind_none_when_out_of_range():
    # Loop size 10, MPI_size 4 -> max index = 9
    # MPI_ind = 3, rank = 3 -> global = 3*4+3 = 15 -> exceeds loop_size
    assert mpiutils.MPI_ind2ind(3, 3, 4, 10) is None


def test_mpi_ind2ind_first_index():
    assert mpiutils.MPI_ind2ind(0, 0, 4, 10) == 0


def test_mpi_ind2ind_last_valid_index():
    # MPI_ind=2, rank=2, size=3, loop_size=9 -> 2*3+2 = 8
    assert mpiutils.MPI_ind2ind(2, 2, 3, 9) == 8


def test_mpi_ind2ind_one_rank():
    # No distribution -> index is the same
    assert mpiutils.MPI_ind2ind(5, 0, 1, 10) == 5
