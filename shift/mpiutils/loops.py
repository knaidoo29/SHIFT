import numpy as np


def get_MPI_loop_size(loop_size, MPI_size):
    """Converts a loop size to the MPI loop size which distributes the loop across
    the cores.

    Parameters
    ----------
    loop_size : int
        Size of the loop.
    MPI_size : int
        Number of processors or MPI jobs.

    Returns
    -------
    MPI_loop_size : int
        Size of the MPI_loop.
    """
    MPI_loop_size = int(np.floor(loop_size/MPI_size))
    if MPI_loop_size*MPI_size < loop_size:
        MPI_loop_size += 1
    return MPI_loop_size


def MPI_ind2ind(MPI_ind, MPI_rank, MPI_size, loop_size):
    """Converts the MPI_ind of a distributed loop to the index of a full loop.

    Parameters
    ----------
    MPI_ind : int
        Index in the MPI loop.
    MPI_rank : int
        The processor assigned.
    MPI_size : int
        Number of processors or MPI jobs.
    loop_size : int
        Size of the loop.

    Returns
    -------
    ind : int
        Index of a full loop.
    """
    ind = MPI_ind*MPI_size + MPI_rank
    if ind >= loop_size:
        ind = None
    return ind
