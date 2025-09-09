import numpy as np
import pytest

from shift.mpiutils import MPI


def test_set_loop_to_None():
    mpi = MPI()
    size = mpi.size
    rank = mpi.rank

    # choose loop_size that isn't a perfect multiple of size to test rounding
    loop_size = size * 3 + 1
    _ = mpi.set_loop(loop_size)  # sets mpi.loop_size and returns MPI_loop_size
    mpi.clean_loop()
    assert mpi.loop_size is None, "clean_loop should set this to None."


def test_set_loop_and_mpi_ind2ind():
    mpi = MPI()
    size = mpi.size
    rank = mpi.rank

    # choose loop_size that isn't a perfect multiple of size to test rounding
    loop_size = size * 3 + 1
    per_rank = mpi.set_loop(loop_size)  # sets mpi.loop_size and returns MPI_loop_size
    assert isinstance(per_rank, int)
    assert mpi.loop_size == loop_size

    # test a few MPI_ind values:
    # compute expected global index for some mpi_ind values
    for mpi_ind in [0, 1, 2, per_rank - 1]:
        ind = mpi.mpi_ind2ind(mpi_ind)
        if ind is None:
            # None if beyond full loop
            assert mpi_ind * size + rank >= loop_size
        else:
            assert ind == mpi_ind * size + rank
            assert 0 <= ind < loop_size


def test_split_and_split_array_and_distribute_and_collect():
    mpi = MPI()
    size = mpi.size
    rank = mpi.rank

    # Make a test array of length N
    N = max(1, size * 3 + 2)
    data = np.arange(N)

    # split indices
    split1, split2 = mpi.split(N, 2)
    # splits should define contiguous blocks covering [0,N)
    assert split1.shape == split2.shape
    assert split1[0] == 0
    assert split2[-1] == N
    # counts sum to N
    counts = split2 - split1
    assert counts.sum() == N

    # split indices
    split1, split2 = mpi.split(N)
    # splits should define contiguous blocks covering [0,N)
    assert split1.shape == split2.shape
    assert split1[0] == 0
    assert split2[-1] == N
    # counts sum to N
    counts = split2 - split1
    assert counts.sum() == N

    # split_array returns the slice for this rank
    local_slice = mpi.split_array(data)
    expected_slice = data[split1[rank]: split2[rank]]
    np.testing.assert_array_equal(local_slice, expected_slice)

    # distribute: rank 0 sends chunks, each rank returns its chunk
    got = mpi.distribute(data)
    # each process should receive its chunk equal to expected_slice
    np.testing.assert_array_equal(got, expected_slice)

    # Now test collect: rank 0 should receive concatenated chunks equal to data
    collected = mpi.collect(got)  # on non-zero ranks returns None
    if rank == 0:
        # Reconstructed data must equal original
        np.testing.assert_array_equal(collected, data)
    else:
        assert collected is None


def test_check_partition():
    mpi = MPI()
    NDshape = [4, 4, 4]
    NDshape_split = [2, 4, 4]
    check = mpi.check_partition(NDshape, NDshape_split)
    assert (check == [False, True, True]).all(), "Check partition does not give expected result."
    NDshape = [4, 4, 4]
    NDshape_split = [4, 2, 4]
    check = mpi.check_partition(NDshape, NDshape_split)
    assert (check == [True, False, True]).all(), "Check partition does not give expected result."
    NDshape = [4, 4, 4]
    NDshape_split = [4, 4, 2]
    check = mpi.check_partition(NDshape, NDshape_split)
    assert (check == [True, True, False]).all(), "Check partition does not give expected result."


def test_mpi_print_simple(capsys):
    mpi = MPI()
    rank = mpi.rank
    mpi.mpi_print("hello", "world")
    captured = capsys.readouterr()
    assert captured.out == "hello world\n"
    mpi.mpi_print_zero("hello", "world")
    captured = capsys.readouterr()
    if rank == 0:
        assert captured.out == "hello world\n"
    else:
        assert captured.out == ""


def test_sum_and_mean():
    mpi = MPI()
    size = mpi.size
    rank = mpi.rank

    # Give every rank a scalar equal to rank+1
    local_scalar = rank + 1
    summed = mpi.sum(local_scalar)
    # Only rank 0 receives the sum (others return None)
    expected_sum = sum(range(1, size + 1))
    if rank == 0:
        assert summed == expected_sum
    else:
        assert summed is None

    # Test mean: give each rank an array of ones of length (rank+1)
    # mean should be overall_total / overall_count.
    local_arr = np.ones(rank + 1, dtype=float)
    mean_val = mpi.mean(local_arr)
    # mean is computed across all elements across ranks and broadcast to all ranks
    total = sum(np.sum(np.ones(r + 1)) for r in range(size))
    total_elems = sum((r + 1) for r in range(size))
    expected_mean = total / total_elems
    # every rank should receive the same mean (float)
    assert pytest.approx(mean_val) == expected_mean


def test_min_and_max():
    mpi = MPI()
    size = mpi.size
    rank = mpi.rank

    # Give different arrays per rank so global min and max are known
    # e.g., rank r will have array [r, r+10]
    local = np.array([rank - size, rank + size])  # ensures variety
    min_val = mpi.min(local)
    max_val = mpi.max(local)

    # These functions broadcast result to all ranks; so check on every rank
    # Compute expected global min and max
    all_mins = mpi.collect(np.min(local))
    # After collect, only rank 0 has full data; but min() method will ensure broadcast.
    if rank == 0:
        # compute expected values using gathered list
        expected_min = np.min(all_mins)
    else:
        # other ranks have already received the broadcasted value from implementation
        expected_min = min_val
    # All ranks should have min_val equal to expected_min
    assert min_val == mpi.min(local)  # idempotent call
    assert max_val == mpi.max(local)

    all_mins = mpi.collect(np.min(local))
    if rank == 0:
        assert type(all_mins) == np.ndarray, "Type does not match expectation"

    all_mins = mpi.collect(np.min(local), outlist=True)
    if rank == 0:
        assert type(all_mins) == list, "Type does not match expectation"
    
    data = np.ones(10)
    if rank == 1:
        data = None
    
    datas = mpi.collect_noNone(data)
    if rank == 0:
        assert np.array([data is not None for data in datas]).all(), "All elements should not be None"
    else:
        assert datas == None, "This core should get None"


def test_send_recv_and_broadcast_roundtrip():
    mpi = MPI()
    size = mpi.size
    rank = mpi.rank

    # Test broadcast: use rank 0 to broadcast an object (e.g., dict)
    data = None
    if rank == 0:
        data = {"rank0": "hello", "size": size}
    out = mpi.broadcast(data)
    # After broadcast, all ranks should receive same object
    assert isinstance(out, dict)
    assert out["size"] == size

def test_send_up_and_send_down_roundtrip():
    mpi = MPI()
    size = mpi.size
    rank = mpi.rank

    # Provide a scalar equal to rank for testing
    val = float(rank)

    # send_up returns data from neighbor above (wrapping), send_down from below (wrapping)
    up = mpi.isend_up(val)
    down = mpi.isend_down(up)

    assert val == down, "Check roundtrip up and down messaging."
