import numpy as np
import scipy.fft as scfft

from typing import Union


def slab_fft1D(
    f_real: np.ndarray, boxsize: float, ngrid: int, axis: int = -1
) -> np.ndarray:
    """
    Performs forward FFT on real space data.

    Parameters
    ----------
    f_real : ndarray
        Real space data.
    boxsize : float
        Box size.
    axis : int, optional
        Axis to perform FFT.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    dx = boxsize / float(ngrid)
    f_fourier = scfft.fft(f_real, axis=axis)
    f_fourier *= dx / np.sqrt(2.0 * np.pi)
    return f_fourier


def slab_ifft1D(
    f_fourier: np.ndarray, boxsize: float, ngrid: int, axis: int = -1
) -> np.ndarray:
    """
    Performs backward FFT on Fourier modes.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float
        Box size.
    axis : int, optional
        Axis to perform FFT.

    Returns
    -------
    f_real : ndarray
        Real space data.
    """
    dx = boxsize / float(ngrid)
    f_real = scfft.ifft(f_fourier, axis=axis)
    f_real *= np.sqrt(2.0 * np.pi) / dx
    return f_real


def slab_dct1D(
    f_real: np.ndarray, boxsize: float, ngrid: int, axis: int = -1, type: int = 2
) -> np.ndarray:
    """
    Performs forward DCT on real space data.

    Parameters
    ----------
    f_real : ndarray
        Input grid data.
    boxsize : float
        Box size.
    axis : int, optional
        Axis to perform FFT.
    type : int, optional
        Type of DCT for scipy to perform.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    dx = boxsize / float(ngrid)
    f_fourier = scfft.dct(f_real, axis=axis, type=type)
    f_fourier *= dx / np.sqrt(2.0 * np.pi)
    return f_fourier


def slab_idct1D(
    f_fourier: np.ndarray, boxsize: float, ngrid: int, axis: int = -1, type: int = 2
) -> np.ndarray:
    """
    Performs backward DCT on Fourier modes.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float
        Box size.
    axis : int, optional
        Axis to perform FFT.
    type : int, optional
        Type of iDCT for scipy to perform.

    Returns
    -------
    f_real : ndarray
        Real space data.
    """
    dx = boxsize / float(ngrid)
    f_real = scfft.idct(f_fourier, axis=axis, type=type)
    f_real *= np.sqrt(2.0 * np.pi) / dx
    return f_real


def slab_dst1D(
    f_real: np.ndarray, boxsize: float, ngrid: int, axis: int = -1, type: int = 2
) -> np.ndarray:
    """
    Performs forward DST on real space data.

    Parameters
    ----------
    f_real : ndarray
        Input grid data.
    boxsize : float
        Box size.
    axis : int, optional
        Axis to perform FFT.
    type : int, optional
        Type of DST for scipy to perform.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    dx = boxsize / float(ngrid)
    f_fourier = scfft.dst(f_real, axis=axis, type=type)
    f_fourier *= dx / np.sqrt(2.0 * np.pi)
    return f_fourier


def slab_idst1D(
    f_fourier: np.ndarray, boxsize: float, ngrid: int, axis: int = -1, type: int = 2
) -> np.ndarray:
    """
    Performs backward DST on Fourier modes.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float
        Box size.
    axis : int, optional
        Axis to perform FFT.
    type : int, optional
        Type of iDST for scipy to perform.

    Returns
    -------
    f_real : ndarray
        Real space data.
    """
    dx = boxsize / float(ngrid)
    f_real = scfft.idst(f_fourier, axis=axis, type=type)
    f_real *= np.sqrt(2.0 * np.pi) / dx
    return f_real


def redistribute_forward_2D(
    f: np.ndarray,
    ngrid: Union[int, list],
    MPI: type,
) -> np.ndarray:
    """
    Redistribute a 2D array from an x-axis slab decomposition to a
    y-axis slab decomposition.

    The input array is assumed to be split across MPI ranks along the
    x-axis, such that each rank stores a local array with shape
    ``(nx_local, Ny)``. The output is redistributed so that each rank
    instead stores a local y-slab with shape ``(Nx, ny_local)``.

    When the x and y dimensions are both evenly divisible by the number
    of MPI ranks, the redistribution is performed with ``Alltoall``.
    Otherwise, ``Alltoallv`` is used to support unequal slab sizes.

    Parameters
    ----------
    f : np.ndarray
        Local input array with shape ``(nx_local, Ny)``.
    ngrid : int or list
        Global grid size. If an integer is supplied, the same grid size
        is assumed along both axes. If a sequence is supplied, it must
        contain ``[Nx, Ny]``.
    MPI : object
        MPIutils MPI object.

    Returns
    -------
    fnew : np.ndarray
        Redistributed local array with shape ``(Nx, ny_local)``.

    Notes
    -----
    The dtype of the input array is preserved.

    For equal slab sizes, the local array is reshaped so that the first
    axis of the communication buffer identifies the destination rank.
    After ``Alltoall``, this axis identifies the source rank, allowing
    the received blocks to be reshaped directly into global x ordering.
    """
    if np.isscalar(ngrid):
        xngrid = yngrid = ngrid
    else:
        if len(ngrid) != 2:
            raise ValueError("Length of grid dimensions must be 2.")
        xngrid, yngrid = ngrid

    xsplits1, xsplits2 = MPI.split(xngrid)
    ysplits1, ysplits2 = MPI.split(yngrid)

    nxs = xsplits2 - xsplits1
    nys = ysplits2 - ysplits1

    nx = nxs[MPI.rank]
    ny = nys[MPI.rank]

    if f.shape != (nx, yngrid):
        raise ValueError(
            f"Expected local shape {(nx, yngrid)}, "
            f"received {f.shape}."
        )

    equal_splits = (
        np.all(nxs == nxs[0])
        and np.all(nys == nys[0])
    )

    if equal_splits:
        # (nx, Ny)
        #   -> (nx, P, ny)
        #   -> (P, nx, ny)
        #
        # The first axis identifies the destination rank.
        sendbuf = np.ascontiguousarray(
            f.reshape(nx, MPI.size, ny)
             .transpose(1, 0, 2)
        )

        recvbuf = np.empty_like(sendbuf)

        MPI.Alltoall(
            sendbuf,
            recvbuf,
        )

        # The first axis now identifies the source rank.
        # Source-rank ordering is the same as global x ordering.
        return recvbuf.reshape(
            xngrid,
            ny,
        )

    # Number of scalar elements sent to each destination rank.
    sendcounts = (
        nx * nys
    ).astype(np.int64)

    senddispls = np.zeros(
        MPI.size,
        dtype=np.int64,
    )
    senddispls[1:] = np.cumsum(
        sendcounts[:-1]
    )

    sendbuf = np.empty(
        int(np.sum(sendcounts)),
        dtype=f.dtype,
    )

    # Pack the y-slab belonging to each destination rank.
    for dest in range(MPI.size):
        start = senddispls[dest]
        stop = start + sendcounts[dest]

        sendbuf[start:stop] = (
            f[
                :,
                ysplits1[dest]:ysplits2[dest],
            ]
            .reshape(-1)
        )

    # This rank receives one x-slab contribution from every source rank.
    recvcounts = (
        nxs * ny
    ).astype(np.int64)

    recvdispls = np.zeros(
        MPI.size,
        dtype=np.int64,
    )
    recvdispls[1:] = np.cumsum(
        recvcounts[:-1]
    )

    recvbuf = np.empty(
        int(np.sum(recvcounts)),
        dtype=f.dtype,
    )

    MPI.Alltoallv(
        sendbuf,
        sendcounts,
        senddispls,
        recvbuf,
        recvcounts,
        recvdispls,
    )

    fnew = np.empty(
        (xngrid, ny),
        dtype=f.dtype,
    )

    # Reconstruct global x ordering from source-rank contributions.
    for source in range(MPI.size):
        start = recvdispls[source]
        stop = start + recvcounts[source]

        fnew[
            xsplits1[source]:xsplits2[source],
            :
        ] = recvbuf[start:stop].reshape(
            nxs[source],
            ny,
        )

    return fnew


def redistribute_forward_3D(
    f: np.ndarray,
    ngrid: Union[int, list],
    MPI: type,
) -> np.ndarray:
    """
    Redistribute a 3D array from an x-axis slab decomposition to a
    y-axis slab decomposition.

    Parameters
    ----------
    f : np.ndarray
        Local array with shape ``(nx_local, Ny, Nz)``.
    ngrid : int or list
        Global grid size along each axis.
    MPI : object
        MPIutils MPI object.

    Returns
    -------
    fnew : np.ndarray
        Redistributed local array with shape ``(Nx, ny_local, Nz)``.
    """
    if np.isscalar(ngrid):
        xngrid = yngrid = zngrid = ngrid
    else:
        if len(ngrid) != 3:
            raise ValueError("Length of grid dimensions must be 3.")
        xngrid, yngrid, zngrid = ngrid

    xsplits1, xsplits2 = MPI.split(xngrid)
    ysplits1, ysplits2 = MPI.split(yngrid)

    nxs = xsplits2 - xsplits1
    nys = ysplits2 - ysplits1

    nx = nxs[MPI.rank]
    ny = nys[MPI.rank]

    if f.shape != (nx, yngrid, zngrid):
        raise ValueError(
            f"Expected local shape {(nx, yngrid, zngrid)}, "
            f"received {f.shape}."
        )

    # Fast path: equal x and y slab widths.
    equal_splits = (
        np.all(nxs == nxs[0])
        and np.all(nys == nys[0])
    )

    if equal_splits:

        # (nx, Ny, Nz)
        #       ↓
        # (nx, P, ny, Nz)
        #       ↓ transpose
        # (P, nx, ny, Nz)
        #
        # First axis now identifies destination rank.
        sendbuf = np.ascontiguousarray(
            f.reshape(nx, MPI.size, ny, zngrid)
             .transpose(1, 0, 2, 3)
        )

        recvbuf = np.empty_like(sendbuf)

        MPI.Alltoall(sendbuf, recvbuf)

        # First axis of recvbuf now identifies source rank.
        # Source-rank ordering is exactly global x ordering.
        return recvbuf.reshape(
            xngrid,
            ny,
            zngrid,
        )

    # General path for unequal slab widths.
    sendcounts = (
        nx * nys * zngrid
    ).astype(np.int64)

    senddispls = np.zeros(MPI.size, dtype=np.int64)
    senddispls[1:] = np.cumsum(sendcounts[:-1])

    sendbuf = np.empty(
        int(np.sum(sendcounts)),
        dtype=f.dtype,
    )

    # Pack each destination y-slab contiguously.
    for dest in range(MPI.size):
        start = senddispls[dest]
        stop = start + sendcounts[dest]

        sendbuf[start:stop] = (
            f[:, ysplits1[dest]:ysplits2[dest], :]
            .reshape(-1)
        )

    # This rank owns ny rows in y after the redistribution.
    # Each source contributes its own x slab.
    recvcounts = (
        nxs * ny * zngrid
    ).astype(np.int64)

    recvdispls = np.zeros(MPI.size, dtype=np.int64)
    recvdispls[1:] = np.cumsum(recvcounts[:-1])

    recvbuf = np.empty(
        int(np.sum(recvcounts)),
        dtype=f.dtype,
    )

    MPI.Alltoallv(
        sendbuf,
        sendcounts,
        senddispls,
        recvbuf,
        recvcounts,
        recvdispls,
    )

    # Reassemble source x-slabs into global x order.
    fnew = np.empty(
        (xngrid, ny, zngrid),
        dtype=f.dtype,
    )

    for source in range(MPI.size):
        start = recvdispls[source]
        stop = start + recvcounts[source]

        block = recvbuf[start:stop].reshape(
            nxs[source],
            ny,
            zngrid,
        )

        fnew[
            xsplits1[source]:xsplits2[source],
            :,
            :
        ] = block

    return fnew



def redistribute_backward_2D(
    f: np.ndarray,
    ngrid: Union[int, list],
    MPI: type,
) -> np.ndarray:
    """
    Redistribute a 2D array from a y-axis slab decomposition back to an
    x-axis slab decomposition.

    The input array is assumed to be split across MPI ranks along the
    y-axis, such that each rank stores a local array with shape
    ``(Nx, ny_local)``. The output is redistributed so that each rank
    instead stores an x-slab with shape ``(nx_local, Ny)``.

    When the x and y dimensions are both evenly divisible by the number
    of MPI ranks, the redistribution is performed with ``Alltoall``.
    Otherwise, ``Alltoallv`` is used to support unequal slab sizes.

    Parameters
    ----------
    f : np.ndarray
        Local input array with shape ``(Nx, ny_local)``.
    ngrid : int or list
        Global grid size. If an integer is supplied, the same grid size
        is assumed along both axes. If a sequence is supplied, it must
        contain ``[Nx, Ny]``.
    MPI : object
        MPIutils MPI object.

    Returns
    -------
    fnew : np.ndarray
        Redistributed local array with shape ``(nx_local, Ny)``.

    Notes
    -----
    The dtype of the input array is preserved.

    This operation is the inverse of ``redistribute_forward_2D``.
    For equal slab sizes, the input is divided into destination x-slabs,
    exchanged with ``Alltoall``, then transposed and reshaped so that
    source-rank ordering reconstructs the global y-axis.
    """
    if np.isscalar(ngrid):
        xngrid = yngrid = ngrid
    else:
        if len(ngrid) != 2:
            raise ValueError("Length of grid dimensions must be 2.")
        xngrid, yngrid = ngrid

    xsplits1, xsplits2 = MPI.split(xngrid)
    ysplits1, ysplits2 = MPI.split(yngrid)

    nxs = xsplits2 - xsplits1
    nys = ysplits2 - ysplits1

    nx = nxs[MPI.rank]
    ny = nys[MPI.rank]

    if f.shape != (xngrid, ny):
        raise ValueError(
            f"Expected local shape {(xngrid, ny)}, "
            f"received {f.shape}."
        )

    equal_splits = (
        np.all(nxs == nxs[0])
        and np.all(nys == nys[0])
    )

    if equal_splits:
        # (Nx, ny)
        #   -> (P, nx, ny)
        #
        # The first axis identifies the destination x-rank.
        sendbuf = np.ascontiguousarray(
            f.reshape(
                MPI.size,
                nx,
                ny,
            )
        )

        recvbuf = np.empty_like(sendbuf)

        MPI.Alltoall(
            sendbuf,
            recvbuf,
        )

        # recvbuf has shape
        # (source_y_rank, nx, ny).
        #
        # Rearrange to
        # (nx, source_y_rank, ny)
        # and collapse the final two axes into global y ordering.
        return np.ascontiguousarray(
            recvbuf.transpose(1, 0, 2)
        ).reshape(
            nx,
            yngrid,
        )

    # Number of scalar elements sent to each destination rank.
    sendcounts = (
        nxs * ny
    ).astype(np.int64)

    senddispls = np.zeros(
        MPI.size,
        dtype=np.int64,
    )
    senddispls[1:] = np.cumsum(
        sendcounts[:-1]
    )

    sendbuf = np.empty(
        int(np.sum(sendcounts)),
        dtype=f.dtype,
    )

    # Pack the x-slab belonging to each destination rank.
    for dest in range(MPI.size):
        start = senddispls[dest]
        stop = start + sendcounts[dest]

        sendbuf[start:stop] = (
            f[
                xsplits1[dest]:xsplits2[dest],
                :
            ]
            .reshape(-1)
        )

    # This rank receives one y-slab contribution from every source rank.
    recvcounts = (
        nx * nys
    ).astype(np.int64)

    recvdispls = np.zeros(
        MPI.size,
        dtype=np.int64,
    )
    recvdispls[1:] = np.cumsum(
        recvcounts[:-1]
    )

    recvbuf = np.empty(
        int(np.sum(recvcounts)),
        dtype=f.dtype,
    )

    MPI.Alltoallv(
        sendbuf,
        sendcounts,
        senddispls,
        recvbuf,
        recvcounts,
        recvdispls,
    )

    fnew = np.empty(
        (nx, yngrid),
        dtype=f.dtype,
    )

    # Reconstruct global y ordering from source-rank contributions.
    for source in range(MPI.size):
        start = recvdispls[source]
        stop = start + recvcounts[source]

        fnew[
            :,
            ysplits1[source]:ysplits2[source],
        ] = recvbuf[start:stop].reshape(
            nx,
            nys[source],
        )

    return fnew


def redistribute_backward_3D(
    f: np.ndarray,
    ngrid: Union[int, list],
    MPI: type,
) -> np.ndarray:
    """
    Redistribute a 3D array from a y-axis slab decomposition back to an
    x-axis slab decomposition.

    Parameters
    ----------
    f : np.ndarray
        Local array with shape ``(Nx, ny_local, Nz)``.
    ngrid : int or list
        Global grid size along each axis.
    MPI : object
        MPIutils MPI object.

    Returns
    -------
    fnew : np.ndarray
        Redistributed local array with shape ``(nx_local, Ny, Nz)``.
    """
    if np.isscalar(ngrid):
        xngrid = yngrid = zngrid = ngrid
    else:
        if len(ngrid) != 3:
            raise ValueError("Length of grid dimensions must be 3.")
        xngrid, yngrid, zngrid = ngrid

    xsplits1, xsplits2 = MPI.split(xngrid)
    ysplits1, ysplits2 = MPI.split(yngrid)

    nxs = xsplits2 - xsplits1
    nys = ysplits2 - ysplits1

    nx = nxs[MPI.rank]
    ny = nys[MPI.rank]

    if f.shape != (xngrid, ny, zngrid):
        raise ValueError(
            f"Expected local shape {(xngrid, ny, zngrid)}, "
            f"received {f.shape}."
        )

    equal_splits = (
        np.all(nxs == nxs[0])
        and np.all(nys == nys[0])
    )

    if equal_splits:

        # (Nx, ny, Nz)
        #       ↓
        # (P, nx, ny, Nz)
        #
        # First axis identifies destination x-rank.
        sendbuf = np.ascontiguousarray(
            f.reshape(
                MPI.size,
                nx,
                ny,
                zngrid,
            )
        )

        recvbuf = np.empty_like(sendbuf)

        MPI.Alltoall(sendbuf, recvbuf)

        # recvbuf:
        # (source_y_rank, nx, ny, Nz)
        #
        # Rearrange to:
        # (nx, source_y_rank, ny, Nz)
        #
        # then collapse source_y_rank × ny -> Ny.
        return np.ascontiguousarray(
            recvbuf.transpose(1, 0, 2, 3)
        ).reshape(
            nx,
            yngrid,
            zngrid,
        )

    # General unequal-slab case.
    sendcounts = (
        nxs * ny * zngrid
    ).astype(np.int64)

    senddispls = np.zeros(MPI.size, dtype=np.int64)
    senddispls[1:] = np.cumsum(sendcounts[:-1])

    sendbuf = np.empty(
        int(np.sum(sendcounts)),
        dtype=f.dtype,
    )

    # Send each destination its x slab.
    for dest in range(MPI.size):
        start = senddispls[dest]
        stop = start + sendcounts[dest]

        sendbuf[start:stop] = (
            f[xsplits1[dest]:xsplits2[dest], :, :]
            .reshape(-1)
        )

    # Every source contributes its y slab to this x slab.
    recvcounts = (
        nx * nys * zngrid
    ).astype(np.int64)

    recvdispls = np.zeros(MPI.size, dtype=np.int64)
    recvdispls[1:] = np.cumsum(recvcounts[:-1])

    recvbuf = np.empty(
        int(np.sum(recvcounts)),
        dtype=f.dtype,
    )

    MPI.Alltoallv(
        sendbuf,
        sendcounts,
        senddispls,
        recvbuf,
        recvcounts,
        recvdispls,
    )

    fnew = np.empty(
        (nx, yngrid, zngrid),
        dtype=f.dtype,
    )

    # Reconstruct global y ordering from source-rank order.
    for source in range(MPI.size):
        start = recvdispls[source]
        stop = start + recvcounts[source]

        block = recvbuf[start:stop].reshape(
            nx,
            nys[source],
            zngrid,
        )

        fnew[
            :,
            ysplits1[source]:ysplits2[source],
            :
        ] = block

    return fnew


def mpi_fft2D(
    f_real: np.ndarray, boxsize: Union[float, list], ngrid: Union[int, list], MPI: type
) -> np.ndarray:
    """
    Performs MPI forward FFT on real space data.

    Parameters
    ----------
    f_real : 1darray or ndarray
        Real space data.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : obj
        MPIutils MPI object.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize = boxsize, boxsize
    else:
        assert len(boxsize) == 2, "Box dimensions must be a list of length 2."
        xboxsize, yboxsize = boxsize[0], boxsize[1]
    if np.isscalar(ngrid):
        xngrid, yngrid = ngrid, ngrid
    else:
        assert len(ngrid) == 2, "grid dimensions must be a list of length 2."
        xngrid, yngrid = ngrid[0], ngrid[1]
    # fft along y
    f_fourier = slab_fft1D(f_real, yboxsize, yngrid, axis=1)
    f_fourier = redistribute_forward_2D(f_fourier, ngrid, MPI)
    # fft along x
    f_fourier = slab_fft1D(f_fourier, xboxsize, xngrid, axis=0)
    return f_fourier


def mpi_ifft2D(
    f_fourier: np.ndarray,
    boxsize: Union[float, list],
    ngrid: Union[int, list],
    MPI: type,
) -> np.ndarray:
    """
    Performs MPI backward FFT on Fourier modes.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : obj
        MPIutils MPI object.

    Returns
    -------
    f_real : 1darray or ndarray
        Real space data.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize = boxsize, boxsize
    else:
        assert len(boxsize) == 2, "Box dimensions must be a list of length 2."
        xboxsize, yboxsize = boxsize[0], boxsize[1]
    if np.isscalar(ngrid):
        xngrid, yngrid = ngrid, ngrid
    else:
        assert len(ngrid) == 2, "grid dimensions must be a list of length 2."
        xngrid, yngrid = ngrid[0], ngrid[1]
    # fft along x
    _f_fourier = slab_ifft1D(f_fourier, xboxsize, xngrid, axis=0)
    _f_fourier = redistribute_backward_2D(_f_fourier, ngrid, MPI)
    # fft along y
    f = slab_ifft1D(_f_fourier, yboxsize, yngrid, axis=1)
    return f.real


def mpi_dct2D(
    f_real: np.ndarray,
    boxsize: Union[float, list],
    ngrid: Union[int, list],
    MPI: type,
    type: int = 2,
) -> np.ndarray:
    """
    Performs MPI forward DCT on real space data.

    Parameters
    ----------
    f_real : 1darray or ndarray
        Real space data.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : obj
        MPIutils MPI object.
    type : int, optional
        Type of DCT for scipy to perform.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize = boxsize, boxsize
    else:
        assert len(boxsize) == 2, "Box dimensions must be a list of length 2."
        xboxsize, yboxsize = boxsize[0], boxsize[1]
    if np.isscalar(ngrid):
        xngrid, yngrid = ngrid, ngrid
    else:
        assert len(ngrid) == 2, "grid dimensions must be a list of length 2."
        xngrid, yngrid = ngrid[0], ngrid[1]
    # fft along y
    f_fourier = slab_dct1D(f_real, yboxsize, yngrid, axis=1, type=type)
    f_fourier = redistribute_forward_2D(f_fourier, ngrid, MPI)
    # fft along x
    f_fourier = slab_dct1D(f_fourier, xboxsize, xngrid, axis=0, type=type)
    return f_fourier


def mpi_idct2D(
    f_fourier: np.ndarray,
    boxsize: Union[float, list],
    ngrid: Union[int, list],
    MPI: type,
    type: int = 2,
) -> np.ndarray:
    """
    Performs MPI backward DCT on Fourier modes.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : obj
        MPIutils MPI object.
    type : int, optional
        Type of iDCT for scipy to perform.

    Returns
    -------
    f_real : 1darray or ndarray
        Real space data.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize = boxsize, boxsize
    else:
        assert len(boxsize) == 2, "Box dimensions must be a list of length 2."
        xboxsize, yboxsize = boxsize[0], boxsize[1]
    if np.isscalar(ngrid):
        xngrid, yngrid = ngrid, ngrid
    else:
        assert len(ngrid) == 2, "grid dimensions must be a list of length 2."
        xngrid, yngrid = ngrid[0], ngrid[1]
    # fft along x
    _f_fourier = slab_idct1D(f_fourier, xboxsize, xngrid, axis=0, type=type)
    _f_fourier = redistribute_backward_2D(_f_fourier, ngrid, MPI)
    # fft along y
    f = slab_idct1D(_f_fourier, yboxsize, yngrid, axis=1, type=type)
    return f


def mpi_dst2D(
    f_real: np.ndarray,
    boxsize: Union[float, list],
    ngrid: Union[int, list],
    MPI: type,
    type: int = 2,
) -> np.ndarray:
    """
    Performs MPI forward DST on real space data.

    Parameters
    ----------
    f_real : 1darray or ndarray
        Real space data.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : obj
        MPIutils MPI object.
    type : int, optional
        Type of DST for scipy to perform.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize = boxsize, boxsize
    else:
        assert len(boxsize) == 2, "Box dimensions must be a list of length 2."
        xboxsize, yboxsize = boxsize[0], boxsize[1]
    if np.isscalar(ngrid):
        xngrid, yngrid = ngrid, ngrid
    else:
        assert len(ngrid) == 2, "grid dimensions must be a list of length 2."
        xngrid, yngrid = ngrid[0], ngrid[1]
    # fft along y
    f_fourier = slab_dst1D(f_real, yboxsize, yngrid, axis=1, type=type)
    f_fourier = redistribute_forward_2D(f_fourier, ngrid, MPI)
    # fft along x
    f_fourier = slab_dst1D(f_fourier, xboxsize, xngrid, axis=0, type=type)
    return f_fourier


def mpi_idst2D(
    f_fourier: np.ndarray,
    boxsize: Union[float, list],
    ngrid: Union[int, list],
    MPI: type,
    type: int = 2,
) -> np.ndarray:
    """
    Performs MPI backward DST on Fourier modes.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : obj
        MPIutils MPI object.
    type : int, optional
        Type of iDST for scipy to perform.

    Returns
    -------
    f_real : 1darray or ndarray
        Real space data.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize = boxsize, boxsize
    else:
        assert len(boxsize) == 2, "Box dimensions must be a list of length 2."
        xboxsize, yboxsize = boxsize[0], boxsize[1]
    if np.isscalar(ngrid):
        xngrid, yngrid = ngrid, ngrid
    else:
        assert len(ngrid) == 2, "grid dimensions must be a list of length 2."
        xngrid, yngrid = ngrid[0], ngrid[1]
    # fft along x
    _f_fourier = slab_idst1D(f_fourier, xboxsize, xngrid, axis=0, type=type)
    _f_fourier = redistribute_backward_2D(_f_fourier, ngrid, MPI)
    # fft along y
    f = slab_idst1D(_f_fourier, yboxsize, yngrid, axis=1, type=type)
    return f


def mpi_fft3D(
    f_real: np.ndarray, boxsize: Union[float, list], ngrid: Union[int, list], MPI: type
) -> np.ndarray:
    """
    Performs MPI forward FFT on real space data.

    Parameters
    ----------
    f_real : ndarray
        Real space data.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : obj
        MPIutils MPI object.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize, zboxsize = boxsize, boxsize, boxsize
    else:
        assert len(boxsize) == 3, "Box dimensions must be a list of length 2."
        xboxsize, yboxsize, zboxsize = boxsize[0], boxsize[1], boxsize[2]
    if np.isscalar(ngrid):
        xngrid, yngrid, zngrid = ngrid, ngrid, ngrid
    else:
        assert len(ngrid) == 3, "grid dimensions must be a list of length 2."
        xngrid, yngrid, zngrid = ngrid[0], ngrid[1], ngrid[2]
    # fft along y
    f_fourier = slab_fft1D(f_real, yboxsize, yngrid, axis=1)
    # fft along z
    f_fourier = slab_fft1D(f_fourier, zboxsize, zngrid, axis=2)
    f_fourier = redistribute_forward_3D(f_fourier, ngrid, MPI)
    # fft along x
    f_fourier = slab_fft1D(f_fourier, xboxsize, xngrid, axis=0)
    return f_fourier


def mpi_ifft3D(
    f_fourier: np.ndarray,
    boxsize: Union[float, list],
    ngrid: Union[int, list],
    MPI: type,
) -> np.ndarray:
    """
    Performs MPI backward FFT on Fourier modes.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : obj
        MPIutils MPI object.

    Returns
    -------
    f : ndarray
        Real space data.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize, zboxsize = boxsize, boxsize, boxsize
    else:
        assert len(boxsize) == 3, "Box dimensions must be a list of length 2."
        xboxsize, yboxsize, zboxsize = boxsize[0], boxsize[1], boxsize[2]
    if np.isscalar(ngrid):
        xngrid, yngrid, zngrid = ngrid, ngrid, ngrid
    else:
        assert len(ngrid) == 3, "grid dimensions must be a list of length 2."
        xngrid, yngrid, zngrid = ngrid[0], ngrid[1], ngrid[2]
    # fft along x
    _f_fourier = slab_ifft1D(f_fourier, xboxsize, xngrid, axis=0)
    _f_fourier = redistribute_backward_3D(_f_fourier, ngrid, MPI)
    # fft along y
    _f_fourier = slab_ifft1D(_f_fourier, yboxsize, yngrid, axis=1)
    # fft along z
    f = slab_ifft1D(_f_fourier, zboxsize, zngrid, axis=2)
    return f.real


def mpi_dct3D(
    f_real: np.ndarray,
    boxsize: Union[float, list],
    ngrid: Union[int, list],
    MPI: type,
    type: int = 2,
) -> np.ndarray:
    """
    Performs MPI forward DCT on real space data.

    Parameters
    ----------
    f_real : ndarray
        Real space data.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : obj
        MPIutils MPI object.
    type : int, optional
        Type of DCT for scipy to perform.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize, zboxsize = boxsize, boxsize, boxsize
    else:
        assert len(boxsize) == 3, "Box dimensions must be a list of length 3."
        xboxsize, yboxsize, zboxsize = boxsize[0], boxsize[1], boxsize[2]
    if np.isscalar(ngrid):
        xngrid, yngrid, zngrid = ngrid, ngrid, ngrid
    else:
        assert len(ngrid) == 3, "grid dimensions must be a list of length 3."
        xngrid, yngrid, zngrid = ngrid[0], ngrid[1], ngrid[2]
    # fft along y
    f_fourier = slab_dct1D(f_real, yboxsize, yngrid, axis=1, type=type)
    # fft along z
    f_fourier = slab_dct1D(f_fourier, zboxsize, zngrid, axis=2, type=type)
    f_fourier = redistribute_forward_3D(f_fourier, ngrid, MPI)
    # fft along x
    f_fourier = slab_dct1D(f_fourier, xboxsize, xngrid, axis=0, type=type)
    return f_fourier


def mpi_idct3D(
    f_fourier: np.ndarray,
    boxsize: Union[float, list],
    ngrid: Union[int, list],
    MPI: type,
    type: int = 2,
) -> np.ndarray:
    """
    Performs MPI backward DCT on Fourier modes.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : obj
        MPIutils MPI object.
    type : int, optional
        Type of iDCT for scipy to perform.

    Returns
    -------
    f : ndarray
        Real space data.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize, zboxsize = boxsize, boxsize, boxsize
    else:
        assert len(boxsize) == 3, "Box dimensions must be a list of length 3."
        xboxsize, yboxsize, zboxsize = boxsize[0], boxsize[1], boxsize[2]
    if np.isscalar(ngrid):
        xngrid, yngrid, zngrid = ngrid, ngrid, ngrid
    else:
        assert len(ngrid) == 3, "grid dimensions must be a list of length 3."
        xngrid, yngrid, zngrid = ngrid[0], ngrid[1], ngrid[2]
    # fft along x
    _f_fourier = slab_idct1D(f_fourier, xboxsize, xngrid, axis=0, type=type)
    _f_fourier = redistribute_backward_3D(_f_fourier, ngrid, MPI)
    # fft along y
    _f_fourier = slab_idct1D(_f_fourier, yboxsize, yngrid, axis=1, type=type)
    # fft along z
    f = slab_idct1D(_f_fourier, zboxsize, zngrid, axis=2, type=type)
    return f


def mpi_dst3D(
    f_real: np.ndarray,
    boxsize: Union[float, list],
    ngrid: Union[int, list],
    MPI: type,
    type: int = 2,
) -> np.ndarray:
    """
    Performs MPI forward DST on real space data.

    Parameters
    ----------
    f_real : ndarray
        Real space data.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : obj
        MPIutils MPI object.
    type : int, optional
        Type of DST for scipy to perform.

    Returns
    -------
    f_fourier : ndarray
        Fourier modes.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize, zboxsize = boxsize, boxsize, boxsize
    else:
        assert len(boxsize) == 3, "Box dimensions must be a list of length 3."
        xboxsize, yboxsize, zboxsize = boxsize[0], boxsize[1], boxsize[2]
    if np.isscalar(ngrid):
        xngrid, yngrid, zngrid = ngrid, ngrid, ngrid
    else:
        assert len(ngrid) == 3, "grid dimensions must be a list of length 3."
        xngrid, yngrid, zngrid = ngrid[0], ngrid[1], ngrid[2]
    # fft along y
    f_fourier = slab_dst1D(f_real, yboxsize, yngrid, axis=1, type=type)
    # fft along z
    f_fourier = slab_dst1D(f_fourier, zboxsize, zngrid, axis=2, type=type)
    f_fourier = redistribute_forward_3D(f_fourier, ngrid, MPI)
    # fft along x
    f_fourier = slab_dst1D(f_fourier, xboxsize, xngrid, axis=0, type=type)
    return f_fourier


def mpi_idst3D(
    f_fourier: np.ndarray,
    boxsize: Union[float, list],
    ngrid: Union[int, list],
    MPI: type,
    type: int = 2,
) -> np.ndarray:
    """
    Performs MPI backward DST on Fourier modes.

    Parameters
    ----------
    f_fourier : ndarray
        Fourier modes.
    boxsize : float or list
        Box size or a list of the dimensions of each axis.
    ngrid : int or list
        Grid division along one axis or a list for each axis.
    MPI : obj
        MPIutils MPI object.
    type : int, optional
        Type of iDST for scipy to perform.

    Returns
    -------
    f : ndarray
        Real space data.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize, zboxsize = boxsize, boxsize, boxsize
    else:
        assert len(boxsize) == 3, "Box dimensions must be a list of length 3."
        xboxsize, yboxsize, zboxsize = boxsize[0], boxsize[1], boxsize[2]
    if np.isscalar(ngrid):
        xngrid, yngrid, zngrid = ngrid, ngrid, ngrid
    else:
        assert len(ngrid) == 3, "grid dimensions must be a list of length 3."
        xngrid, yngrid, zngrid = ngrid[0], ngrid[1], ngrid[2]
    # fft along x
    _f_fourier = slab_idst1D(f_fourier, xboxsize, xngrid, axis=0, type=type)
    _f_fourier = redistribute_backward_3D(_f_fourier, ngrid, MPI)
    # fft along y
    _f_fourier = slab_idst1D(_f_fourier, yboxsize, yngrid, axis=1, type=type)
    # fft along z
    f = slab_idst1D(_f_fourier, zboxsize, zngrid, axis=2, type=type)
    return f
