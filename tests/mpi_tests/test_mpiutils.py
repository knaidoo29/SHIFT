"""MPI regression tests for SHIFT's MPI wrapper.

Run with, e.g.:
    mpiexec -n 1 python -m pytest -q tests/test_mpiutils.py
    mpiexec -n 4 python -m pytest -q tests/test_mpiutils.py

All ranks must run the *same* tests in the same order.  Do not use pytest-xdist.
Tests of MPI.Finalize (MPI.end) are intentionally omitted.
"""

import numpy as np
import pytest

from shift.mpiutils import MPI
from shift.mpiutils.mpiclass import get_MPI_loop_size


@pytest.fixture(scope="module")
def mpi():
    return MPI()


def _structured_dtype(precision):
    """Create a structured NumPy dtype containing vector, integer, and scalar fields."""
    return np.dtype([
        ("vector", precision, (3,)),
        ("identifier", np.uint64),
        ("scalar", precision),
    ])


def _structured_array(rank, n, precision=np.float32):
    """Generate a structured array with reproducible values for MPI testing."""
    data = np.zeros(n, dtype=_structured_dtype(precision))
    data["identifier"] = rank * 100_000 + np.arange(n)
    data["vector"] = rank + np.arange(n)[:, None]
    data["scalar"] = rank
    return data


def _routed_rows(source, size, *, ndim=1, empty_odd=False):
    """Reference input to Alltoall_arrays, independently rebuildable on any rank."""
    n = 0 if empty_odd and source % 2 else source + 3
    i = np.arange(n, dtype=np.int64)
    dest = (i + source) % size
    if ndim == 1:
        data = 10_000 * source + i
    else:
        data = np.column_stack((10_000 * source + i, -10_000 * source - i))
    return data, dest.astype(np.int64)


# ---------------------------------------------------------------------------
# Local helpers, printing and synchronization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length,parts,expected", [
    (0, 4, 0),
    (1, 4, 1),
    (8, 4, 2),
    (9, 4, 3),
    (17, 5, 4),
])
def test_get_mpi_loop_size(length, parts, expected):
    assert get_MPI_loop_size(length, parts) == expected


def test_mpi_initialization_and_loop(mpi):
    assert 0 <= mpi.rank < mpi.size
    assert mpi.size >= 1
    assert mpi.loop_size is None
    assert mpi.mpi_info == f"Proc {mpi.rank + 1} of {mpi.size}"

    n = 3 * mpi.size + 1
    assert mpi.set_loop(n) == 4
    assert mpi.loop_size == n
    mpi.clean_loop()
    assert mpi.loop_size is None


def test_barrier(mpi):
    assert mpi.Barrier() is None


@pytest.mark.parametrize("nparts", [None, 2, 5])
@pytest.mark.parametrize("n", [0, 1, 2, 11, 37])
def test_split(mpi, n, nparts):
    starts, ends = mpi.split(n, nparts)
    p = mpi.size if nparts is None else nparts

    assert starts.shape == (p,)
    assert ends.shape == (p,)
    assert starts[0] == 0
    assert ends[-1] == n
    np.testing.assert_array_equal(starts[1:], ends[:-1])
    counts = ends - starts
    assert np.all(counts >= 0)
    assert counts.sum() == n
    assert counts.max() - counts.min() <= 1
    np.testing.assert_array_equal(
        counts, np.array([n // p + (r < n % p) for r in range(p)])
    )


def test_split_invalid(mpi):
    with pytest.raises(ValueError):
        mpi.split(-1)
    with pytest.raises(ValueError):
        mpi.split(3, size=0)


def test_split_array(mpi):
    data = np.arange(3 * mpi.size + 2).reshape(-1, 1)
    starts, ends = mpi.split(len(data))
    np.testing.assert_array_equal(
        mpi.split_array(data), data[starts[mpi.rank]:ends[mpi.rank]]
    )


def test_mpi_print(mpi, capsys):
    mpi.mpi_print("hello", "world")
    assert capsys.readouterr().out == "hello world\n"

    mpi.mpi_print_zero("root", 42)
    assert capsys.readouterr().out == ("root 42\n" if mpi.rank == 0 else "")


# ---------------------------------------------------------------------------
# Object and buffer broadcasts / gathers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("root_selector", ["first", "last"])
def test_bcast_python_object(mpi, root_selector):
    root = 0 if root_selector == "first" else mpi.size - 1
    payload = {"message": "hello", "size": mpi.size, "items": [1, None, 3]}
    sent = payload if mpi.rank == root else None
    assert mpi.bcast(sent, root=root) == payload


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_Bcast_numpy_buffer(mpi, dtype):
    root = mpi.size - 1
    expected = np.arange(12, dtype=dtype).reshape(3, 4)
    data = expected.copy() if mpi.rank == root else np.empty_like(expected)
    assert mpi.Bcast(data, root=root) is None
    np.testing.assert_array_equal(data, expected)


def test_Bcast_structured_records(mpi):
    expected = _structured_array(0, 3)
    data = expected.copy() if mpi.rank == 0 else np.empty_like(expected)

    mpi.Bcast(data)

    np.testing.assert_array_equal(data, expected)


def test_gather_python_objects(mpi):
    record = {"rank": mpi.rank, "values": [mpi.rank, mpi.rank + 1]}
    result = mpi.gather(record, root=0)
    if mpi.rank == 0:
        assert isinstance(result, list)
        assert result == [
            {"rank": r, "values": [r, r + 1]}
            for r in range(mpi.size)
        ]
    else:
        assert result is None


def test_gather_preserves_array_boundaries(mpi):
    # Regression for PEPSI printing a scalar instead of a row of diagnostics.
    record = np.array([mpi.rank, 100 + mpi.rank, 200 + mpi.rank], dtype=np.int64)
    result = mpi.gather(record, root=0)
    if mpi.rank == 0:
        assert isinstance(result, list)
        assert len(result) == mpi.size
        for r, row in enumerate(result):
            np.testing.assert_array_equal(row, [r, 100 + r, 200 + r])
    else:
        assert result is None


@pytest.mark.parametrize("mode", ["equal", "unequal", "some_empty"])
@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_Gather_numeric(mpi, mode, dtype):
    root = mpi.size - 1
    n = {
        "equal": lambda r: 3,
        "unequal": lambda r: r + 1,
        "some_empty": lambda r: 0 if r % 2 else r + 2,
    }[mode]
    local = (100 * mpi.rank + np.arange(n(mpi.rank))).astype(dtype)
    gathered = mpi.Gather(local, root=root)
    if mpi.rank == root:
        expected = np.concatenate([
            (100 * r + np.arange(n(r))).astype(dtype)
            for r in range(mpi.size)
        ])
        assert isinstance(gathered, np.ndarray)
        assert gathered.dtype == np.dtype(dtype)
        assert gathered.ndim == 1  # Buffer Gather intentionally concatenates.
        np.testing.assert_array_equal(gathered, expected)
    else:
        assert gathered is None


@pytest.mark.parametrize("n_mode", ["even", "uneven", "smaller_than_ranks", "empty"])
@pytest.mark.parametrize("dtype", [np.float32, np.uint64])
def test_Scatter_numeric(mpi, n_mode, dtype):
    root = mpi.size - 1
    n = {
        "even": mpi.size * 3,
        "uneven": mpi.size * 3 + 1,
        "smaller_than_ranks": max(0, mpi.size - 1),
        "empty": 0,
    }[n_mode]
    data = (np.arange(n, dtype=dtype) + 7) if mpi.rank == root else None
    got = mpi.Scatter(data, dtype=dtype, root=root)
    starts, ends = mpi.split(n)
    expected = (np.arange(n, dtype=dtype) + 7)[starts[mpi.rank]:ends[mpi.rank]]
    assert got.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(got, expected)


def test_Scatter_Gather_roundtrip(mpi):
    n = mpi.size * 4 + 1
    original = np.arange(n, dtype=np.float64)
    local = mpi.Scatter(original if mpi.rank == 0 else None, dtype=np.float64)
    result = mpi.Gather(local)
    if mpi.rank == 0:
        np.testing.assert_array_equal(result, original)
    else:
        assert result is None


# ---------------------------------------------------------------------------
# Periodic neighbor exchanges, with both numeric and structured arrays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint64])
def test_Send_up_and_down_fixed_size(mpi, dtype):
    local = (10 * mpi.rank + np.arange(12)).astype(dtype).reshape(3, 4)
    up = mpi.Send_up(local, tag=100)
    down = mpi.Send_down(local, tag=110)
    prev = (mpi.rank - 1) % mpi.size
    nxt = (mpi.rank + 1) % mpi.size
    np.testing.assert_array_equal(
        up, (10 * prev + np.arange(12)).astype(dtype).reshape(3, 4)
    )
    np.testing.assert_array_equal(
        down, (10 * nxt + np.arange(12)).astype(dtype).reshape(3, 4)
    )
    np.testing.assert_array_equal(mpi.Send_down(up, tag=120), local)
    np.testing.assert_array_equal(mpi.Send_up(down, tag=130), local)


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_Send_up_and_down_structured(mpi, precision):
    local = _structured_array(mpi.rank, 4, precision)
    prev = (mpi.rank - 1) % mpi.size
    nxt = (mpi.rank + 1) % mpi.size
    up = mpi.Send_up(local, tag=140)
    down = mpi.Send_down(local, tag=150)
    assert up.dtype == local.dtype
    assert down.dtype == local.dtype
    np.testing.assert_array_equal(up, _structured_array(prev, 4, precision))
    np.testing.assert_array_equal(down, _structured_array(nxt, 4, precision))


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("dtype", [np.float32, np.uint64])
def test_Send_variable_numeric(mpi, direction, dtype):
    # Rank 1 (if present) sends an empty array, other ranks send different lengths.
    n = lambda r: 0 if (r % 3 == 1) else r + 2
    local = (100 * mpi.rank + np.arange(n(mpi.rank))).astype(dtype).reshape(-1, 1)
    if direction == "up":
        got = mpi.Send_up_variable(local, tag=160)
        src = (mpi.rank - 1) % mpi.size
    else:
        got = mpi.Send_down_variable(local, tag=170)
        src = (mpi.rank + 1) % mpi.size
    expected = (100 * src + np.arange(n(src))).astype(dtype).reshape(-1, 1)
    assert got.dtype == local.dtype
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_Send_variable_structured(mpi, direction, precision):
    n = lambda r: 0 if (r % 3 == 1) else r + 2
    local = _structured_array(mpi.rank, n(mpi.rank), precision)
    if direction == "up":
        got = mpi.Send_up_variable(local, tag=180)
        src = (mpi.rank - 1) % mpi.size
    else:
        got = mpi.Send_down_variable(local, tag=190)
        src = (mpi.rank + 1) % mpi.size
    expected = _structured_array(src, n(src), precision)
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_Send_variable_noncontiguous_source(mpi):
    base = (100 * mpi.rank + np.arange(2 * (mpi.rank + 2))).astype(np.float64)
    local = base[::2]
    assert not local.flags.c_contiguous or local.size <= 1
    src = (mpi.rank - 1) % mpi.size
    got = mpi.Send_up_variable(local, tag=200)
    expected = (100 * src + np.arange(2 * (src + 2)))[::2].astype(np.float64)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# Buffer all-to-all operations
# ---------------------------------------------------------------------------


def test_Alltoall_equal_numeric(mpi):
    # Sender s sends [100*s + 10*d + 0, 100*s + 10*d + 1] to destination d.
    send = np.array(
        [[100 * mpi.rank + 10 * d, 100 * mpi.rank + 10 * d + 1]
         for d in range(mpi.size)], dtype=np.int64
    )
    recv = np.empty_like(send)
    assert mpi.Alltoall(send, recv) is None
    expected = np.array(
        [[100 * s + 10 * mpi.rank, 100 * s + 10 * mpi.rank + 1]
         for s in range(mpi.size)], dtype=np.int64
    )
    np.testing.assert_array_equal(recv, expected)


def test_Alltoallv_variable_numeric(mpi):
    # Distinct message lengths (including zero for some non-self pairs).
    count = lambda s, d: (
        0 if s != d and (s + d) % 3 == 0 else 1 + (2 * s + d) % 3
    )
    sendcounts = np.array([count(mpi.rank, d) for d in range(mpi.size)], dtype=np.int64)
    recvcounts = np.array([count(s, mpi.rank) for s in range(mpi.size)], dtype=np.int64)
    senddispls = np.r_[0, np.cumsum(sendcounts[:-1])].astype(np.int64)
    recvdispls = np.r_[0, np.cumsum(recvcounts[:-1])].astype(np.int64)
    send = np.concatenate([
        1000 * mpi.rank + 100 * d + np.arange(count(mpi.rank, d))
        for d in range(mpi.size)
    ]).astype(np.int64)
    recv = np.empty(int(recvcounts.sum()), dtype=np.int64)

    assert mpi.Alltoallv(send, sendcounts, senddispls,
                         recv, recvcounts, recvdispls) is None
    expected = np.concatenate([
        1000 * s + 100 * mpi.rank + np.arange(count(s, mpi.rank))
        for s in range(mpi.size)
    ]).astype(np.int64)
    np.testing.assert_array_equal(recv, expected)


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("empty_odd", [False, True])
def test_Alltoall_arrays(mpi, ndim, empty_odd):
    local, targets = _routed_rows(mpi.rank, mpi.size, ndim=ndim, empty_odd=empty_odd)
    got = mpi.Alltoall_arrays(local, targets)
    expected_pieces = []
    for source in range(mpi.size):
        data, ranks = _routed_rows(source, mpi.size, ndim=ndim, empty_odd=empty_odd)
        expected_pieces.append(data[ranks == mpi.rank])
    expected = np.concatenate(expected_pieces, axis=0)
    assert got.dtype == local.dtype
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_Alltoall_arrays_invalid_local_arguments(mpi):
    # Only pre-MPI validation is tested: every rank follows the same error path.
    data = np.arange(3)
    with pytest.raises(ValueError, match="one-dimensional"):
        mpi.Alltoall_arrays(data, np.zeros((3, 1), dtype=int))
    with pytest.raises(ValueError, match="match data.shape"):
        mpi.Alltoall_arrays(data, np.zeros(2, dtype=int))
    with pytest.raises(TypeError, match="integer"):
        mpi.Alltoall_arrays(data, np.zeros(3, dtype=float))
    with pytest.raises(ValueError, match="out of range"):
        mpi.Alltoall_arrays(data, np.full(3, mpi.size, dtype=int))


# ---------------------------------------------------------------------------
# Reductions and precision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("root_selector", ["first", "last"])
def test_Sum_scalar(mpi, root_selector):
    root = 0 if root_selector == "first" else mpi.size - 1
    result = mpi.Sum(mpi.rank + 1, root=root)
    if mpi.rank == root:
        assert result == mpi.size * (mpi.size + 1) // 2
    else:
        assert result is None


def test_Sum_array(mpi):
    local = np.array([mpi.rank, 1, 2 * mpi.rank], dtype=np.float64)
    result = mpi.Sum(local)
    if mpi.rank == 0:
        s = mpi.size * (mpi.size - 1) // 2
        np.testing.assert_array_equal(result, [s, mpi.size, 2 * s])
    else:
        assert result is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Mean_float(mpi, dtype):
    # Varying local lengths, with rank 1 empty.
    local = np.full(0 if mpi.rank == 1 else mpi.rank + 1,
                    mpi.rank + 0.5, dtype=dtype)
    got = mpi.Mean(local)
    numerator = sum((0 if r == 1 else r + 1) * (r + 0.5)
                    for r in range(mpi.size))
    denominator = sum(0 if r == 1 else r + 1 for r in range(mpi.size))
    assert got == pytest.approx(numerator / denominator, rel=1e-6)
    assert np.asarray(got).dtype == np.dtype(dtype)


def test_Mean_integer_promotes_to_float64(mpi):
    local = np.array([mpi.rank, mpi.rank + 2], dtype=np.int64)
    got = mpi.Mean(local)
    assert got == pytest.approx((mpi.size + 1) / 2)
    assert np.asarray(got).dtype == np.dtype(np.float64)


def test_Mean_float16_promotes_to_float32(mpi):
    local = np.array([mpi.rank + 1], dtype=np.float16)
    got = mpi.Mean(local)
    assert got == pytest.approx((mpi.size + 1) / 2)
    assert np.asarray(got).dtype == np.dtype(np.float32)


def test_Mean_empty_global_raises(mpi):
    with pytest.raises(ValueError, match="empty distributed"):
        mpi.Mean(np.empty(0, dtype=np.float64))


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float32, np.bool_])
def test_Min_Max_with_some_empty_ranks(mpi, dtype):
    # Rank 1 is empty if present. Other ranks contribute two elements.
    if mpi.rank == 1:
        local = np.empty(0, dtype=dtype)
    elif dtype is np.bool_:
        local = np.array([mpi.rank % 2 == 0, True], dtype=dtype)
    else:
        local = np.array([mpi.rank, mpi.rank + 10], dtype=dtype)

    inputs = []
    for r in range(mpi.size):
        if r == 1:
            continue
        if dtype is np.bool_:
            inputs.extend([r % 2 == 0, True])
        else:
            inputs.extend([r, r + 10])

    assert mpi.Min(local) == min(inputs)
    assert mpi.Max(local) == max(inputs)


# ---------------------------------------------------------------------------
# Known structured-buffer limitations
# ---------------------------------------------------------------------------


def test_structured_Scatter(mpi):
    dtype = _structured_dtype(np.float32)
    n = 3 * mpi.size + 1
    local = mpi.Scatter(_structured_array(0, n) if mpi.rank == 0 else None, dtype=dtype)
    starts, ends = mpi.split(n)
    np.testing.assert_array_equal(local, _structured_array(0, n)[starts[mpi.rank]:ends[mpi.rank]])


def test_structured_Gather(mpi):
    local = _structured_array(mpi.rank, mpi.rank + 1)
    got = mpi.Gather(local)
    if mpi.rank == 0:
        expected = np.concatenate([_structured_array(r, r + 1) for r in range(mpi.size)])
        np.testing.assert_array_equal(got, expected)
    else:
        assert got is None


def test_structured_Alltoall_arrays(mpi):
    data = _structured_array(mpi.rank, mpi.size + 2)
    targets = np.arange(len(data), dtype=np.int64) % mpi.size
    got = mpi.Alltoall_arrays(data, targets)
    expected = np.concatenate([
        _structured_array(r, mpi.size + 2)[targets == mpi.rank]
        for r in range(mpi.size)
    ])
    np.testing.assert_array_equal(got, expected)
