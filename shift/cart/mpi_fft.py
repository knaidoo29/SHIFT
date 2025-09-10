import numpy as np
import scipy.fft as scfft

from typing import Tuple, Union


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


def _get_splits_subset_2D(
    rank1: int,
    rank2: int,
    xsplits1: np.ndarray,
    xsplits2: np.ndarray,
    ysplits1: np.ndarray,
    ysplits2: np.ndarray,
    reverse: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Internal function for finding index splits across two axes based on input
    node ranks.

    Parameters
    ----------
    rank1, rank2 : int
        Node ranks.
    xsplits1, xsplits2, ysplits1, ysplits2 : int array
        Arrays showing the split index for splitting x and y axis across nodes.
    """
    if reverse == True:
        return xsplits1[rank1], xsplits2[rank1], ysplits1[rank2], ysplits2[rank2]
    else:
        return xsplits1[rank2], xsplits2[rank2], ysplits1[rank1], ysplits2[rank1]


def _get_splits_2D(
    MPI: type, xngrid: int, yngrid: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the beginning and end index for splitting arrays in 2 dimensions.

    Parameters
    ----------
    MPI : function
        MPIutils MPI object.
    xngrid, yngrid : int
        Grid size along each axis.

    Returns
    -------
    xsplits1, xsplits2, ysplits1, ysplits2 : int array
        Arrays showing the split index for splitting x and y axis across nodes.
    """
    xsplits1, xsplits2 = MPI.split(xngrid)
    ysplits1, ysplits2 = MPI.split(yngrid)
    return xsplits1, xsplits2, ysplits1, ysplits2


def _get_splits_3D(
    MPI: type, xngrid: int, yngrid: int, zngrid: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the beginning and end index for splitting arrays in 3 dimensions.

    Parameters
    ----------
    MPI : function
        MPIutils MPI object.
    xngrid, yngrid, zngrid : int
        Grid size along each axis.

    Returns
    -------
    xsplits1, xsplits2, ysplits1, ysplits2, zsplits1, zsplits2 : int array
        Arrays showing the split index for splitting x, y and z axis across nodes.
    """
    xsplits1, xsplits2 = MPI.split(xngrid)
    ysplits1, ysplits2 = MPI.split(yngrid)
    zsplits1, zsplits2 = MPI.split(zngrid)
    return xsplits1, xsplits2, ysplits1, ysplits2, zsplits1, zsplits2


def _get_empty_split_array_2D(
    xsplits1: np.ndarray,
    xsplits2: np.ndarray,
    ysplits1: np.ndarray,
    ysplits2: np.ndarray,
    rank: int,
    axis: int = 0,
    iscomplex: bool = False,
) -> np.ndarray:
    """
    Returns a normal or complex array with zeros.

    Parameters
    ----------
    xsplits1, xsplits2, ysplits1, ysplits2 : int array
        Arrays showing the split index for splitting x and y axis across nodes.
    rank : int
        MPI node rank.
    axis : int, optional
        Axis where the array is being split across.
    iscomplex : bool, optional
        Is the output array complex.
    """
    assert axis == 0 or axis == 1, "axis %i is unsupported for 2D grid" % axis
    if axis == 0:
        xsplit1, xsplit2 = xsplits1[rank], xsplits2[rank]
        ysplit1, ysplit2 = ysplits1[0], ysplits2[-1]
    else:
        xsplit1, xsplit2 = xsplits1[0], xsplits2[-1]
        ysplit1, ysplit2 = ysplits1[rank], ysplits2[rank]
    if iscomplex:
        return np.zeros((xsplit2 - xsplit1, ysplit2 - ysplit1)) + 1j * np.zeros(
            (xsplit2 - xsplit1, ysplit2 - ysplit1)
        )
    else:
        return np.zeros((xsplit2 - xsplit1, ysplit2 - ysplit1))


def _get_empty_split_array_3D(
    xsplits1: np.ndarray,
    xsplits2: np.ndarray,
    ysplits1: np.ndarray,
    ysplits2: np.ndarray,
    zsplits1: np.ndarray,
    zsplits2: np.ndarray,
    rank: int,
    axis: int = 0,
    iscomplex: bool = False,
) -> np.ndarray:
    """
    Returns a normal or complex array with zeros.

    Parameters
    ----------
    xsplits1, xsplits2, ysplits1, ysplits2, zsplits1, zsplits2 : int array
        Arrays showing the split index for splitting x, y and z axis across nodes.
    rank : int
        MPI node rank.
    axis : int, optional
        Axis where the array is being split across.
    iscomplex : bool, optional
        Is the output array complex.
    """
    assert axis == 0 or axis == 1, "axis %i is unsupported for 3D grid" % axis
    if axis == 0:
        xsplit1, xsplit2 = xsplits1[rank], xsplits2[rank]
        ysplit1, ysplit2 = ysplits1[0], ysplits2[-1]
        zsplit1, zsplit2 = zsplits1[0], zsplits2[-1]
    else:
        xsplit1, xsplit2 = xsplits1[0], xsplits2[-1]
        ysplit1, ysplit2 = ysplits1[rank], ysplits2[rank]
        zsplit1, zsplit2 = zsplits1[0], zsplits2[-1]
    if iscomplex:
        return np.zeros(
            (xsplit2 - xsplit1, ysplit2 - ysplit1, zsplit2 - zsplit1)
        ) + 1j * np.zeros((xsplit2 - xsplit1, ysplit2 - ysplit1, zsplit2 - zsplit1))
    else:
        return np.zeros((xsplit2 - xsplit1, ysplit2 - ysplit1, zsplit2 - zsplit1))


def redistribute_forward_2D(
    f: np.ndarray, ngrid: Union[int, list], MPI: type, iscomplex: bool = False
) -> np.ndarray:
    """
    Redistributes a 2D array from the conventional axis split across x to a
    split across y.

    Parameters
    ----------
    f : 2darray
        Input data split across the x axis.
    ngrid : int or list
        Grid size along each axis or a list of grid sizes along each axes.
    MPI : object
        MPIutils MPI object.
    iscomplex : bool, optional
        Is the input data complex.
    """
    if np.isscalar(ngrid):
        xngrid, yngrid = ngrid, ngrid
    else:
        assert len(ngrid) == 2, "Length of grid dimensions must be 2."
        xngrid, yngrid = ngrid[0], ngrid[1]
    _xsplits1, _xsplits2, _ysplits1, _ysplits2 = _get_splits_2D(MPI, xngrid, yngrid)
    for i in range(0, MPI.size):
        if i == MPI.rank:
            fnew = _get_empty_split_array_2D(
                _xsplits1,
                _xsplits2,
                _ysplits1,
                _ysplits2,
                MPI.rank,
                axis=1,
                iscomplex=iscomplex,
            )
            for j in range(0, MPI.size):
                if j != MPI.rank:
                    _f = MPI.recv(j, tag=i * MPI.size + j)
                else:
                    _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(
                        MPI.rank,
                        i,
                        _xsplits1,
                        _xsplits2,
                        _ysplits1,
                        _ysplits2,
                        reverse=True,
                    )
                    _f = f[:, _ysplit1:_ysplit2]
                _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(
                    j,
                    MPI.rank,
                    _xsplits1,
                    _xsplits2,
                    _ysplits1,
                    _ysplits2,
                    reverse=True,
                )
                fnew[_xsplit1:_xsplit2, :] = _f
            fnew = np.array(fnew)
        else:
            _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(
                MPI.rank, i, _xsplits1, _xsplits2, _ysplits1, _ysplits2, reverse=True
            )
            MPI.send(f[:, _ysplit1:_ysplit2], to_rank=i, tag=i * MPI.size + MPI.rank)
        MPI.wait()
    return fnew


def redistribute_forward_3D(
    f: np.ndarray, ngrid: Union[int, list], MPI: type, iscomplex: bool = False
) -> np.ndarray:
    """
    Redistributes a 3D array from the conventional axis split across x to a
    split across y.

    Parameters
    ----------
    f : 3darray
        Input data split across the x axis.
    ngrid : int or list
        Grid size along each axis or a list of grid sizes along each axes.
    MPI : object
        MPIutils MPI object.
    iscomplex : bool, optional
        Is the input data complex.
    """
    if np.isscalar(ngrid):
        xngrid, yngrid, zngrid = ngrid, ngrid, ngrid
    else:
        assert len(ngrid) == 3, "Length of grid dimensions must be 3."
        xngrid, yngrid, zngrid = ngrid[0], ngrid[1], ngrid[2]
    _xsplits1, _xsplits2, _ysplits1, _ysplits2, _zsplits1, _zsplits2 = _get_splits_3D(
        MPI, xngrid, yngrid, zngrid
    )
    for i in range(0, MPI.size):
        if i == MPI.rank:
            fnew = _get_empty_split_array_3D(
                _xsplits1,
                _xsplits2,
                _ysplits1,
                _ysplits2,
                _zsplits1,
                _zsplits2,
                MPI.rank,
                axis=1,
                iscomplex=iscomplex,
            )
            for j in range(0, MPI.size):
                if j != MPI.rank:
                    _f = MPI.recv(j, tag=i * MPI.size + j)
                else:
                    _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(
                        MPI.rank,
                        i,
                        _xsplits1,
                        _xsplits2,
                        _ysplits1,
                        _ysplits2,
                        reverse=True,
                    )
                    _f = f[:, _ysplit1:_ysplit2, :]
                _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(
                    j,
                    MPI.rank,
                    _xsplits1,
                    _xsplits2,
                    _ysplits1,
                    _ysplits2,
                    reverse=True,
                )
                fnew[_xsplit1:_xsplit2, :, :] = _f
            fnew = np.array(fnew)
        else:
            _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(
                MPI.rank, i, _xsplits1, _xsplits2, _ysplits1, _ysplits2, reverse=True
            )
            MPI.send(f[:, _ysplit1:_ysplit2, :], to_rank=i, tag=i * MPI.size + MPI.rank)
        MPI.wait()
    return fnew


def redistribute_backward_2D(
    f: np.ndarray, ngrid: Union[int, list], MPI: type, iscomplex: int = False
) -> np.ndarray:
    """
    Redistributes a 2D array backwards a split across y to a split across x.

    Parameters
    ----------
    f : 2darray
        Input data split across the x axis.
    ngrid : int or list
        Grid size along each axis or a list of grid sizes along each axes.
    MPI : object
        MPIutils MPI object.
    iscomplex : bool, optional
        Is the input data complex.
    """
    if np.isscalar(ngrid):
        xngrid, yngrid = ngrid, ngrid
    else:
        assert len(ngrid) == 2, "Length of grid dimensions must be 2."
        xngrid, yngrid = ngrid[0], ngrid[1]
    _xsplits1, _xsplits2, _ysplits1, _ysplits2 = _get_splits_2D(MPI, xngrid, yngrid)
    for i in range(0, MPI.size):
        if i == MPI.rank:
            fnew = _get_empty_split_array_2D(
                _xsplits1,
                _xsplits2,
                _ysplits1,
                _ysplits2,
                MPI.rank,
                axis=0,
                iscomplex=iscomplex,
            )
            for j in range(0, MPI.size):
                if j != MPI.rank:
                    _f = MPI.recv(j, tag=i * MPI.size + j)
                else:
                    _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(
                        MPI.rank,
                        i,
                        _xsplits1,
                        _xsplits2,
                        _ysplits1,
                        _ysplits2,
                        reverse=False,
                    )
                    _f = f[_xsplit1:_xsplit2, :]
                _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(
                    j,
                    MPI.rank,
                    _xsplits1,
                    _xsplits2,
                    _ysplits1,
                    _ysplits2,
                    reverse=False,
                )
                fnew[:, _ysplit1:_ysplit2] = _f
            fnew = np.array(fnew)
        else:
            _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(
                MPI.rank, i, _xsplits1, _xsplits2, _ysplits1, _ysplits2, reverse=False
            )
            MPI.send(f[_xsplit1:_xsplit2, :], to_rank=i, tag=i * MPI.size + MPI.rank)
        MPI.wait()
    return fnew


def redistribute_backward_3D(
    f: np.ndarray, ngrid: Union[int, list], MPI: type, iscomplex: bool = False
) -> np.ndarray:
    """
    Redistributes a 3D array backwards a split across y to a split across x.

    Parameters
    ----------
    f : 3darray
        Input data split across the x axis.
    ngrid : int or list
        Grid size along each axis or a list of grid sizes along each axes.
    MPI : object
        MPIutils MPI object.
    iscomplex : bool, optional
        Is the input data complex.
    """
    if np.isscalar(ngrid):
        xngrid, yngrid, zngrid = ngrid, ngrid, ngrid
    else:
        assert len(ngrid) == 3, "Length of grid dimensions must be 3."
        xngrid, yngrid, zngrid = ngrid[0], ngrid[1], ngrid[2]
    _xsplits1, _xsplits2, _ysplits1, _ysplits2, _zsplits1, _zsplits2 = _get_splits_3D(
        MPI, xngrid, yngrid, zngrid
    )
    for i in range(0, MPI.size):
        if i == MPI.rank:
            fnew = _get_empty_split_array_3D(
                _xsplits1,
                _xsplits2,
                _ysplits1,
                _ysplits2,
                _zsplits1,
                _zsplits2,
                MPI.rank,
                axis=0,
                iscomplex=iscomplex,
            )
            for j in range(0, MPI.size):
                if j != MPI.rank:
                    _f = MPI.recv(j, tag=i * MPI.size + j)
                else:
                    _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(
                        MPI.rank,
                        i,
                        _xsplits1,
                        _xsplits2,
                        _ysplits1,
                        _ysplits2,
                        reverse=False,
                    )
                    _f = f[_xsplit1:_xsplit2, :, :]
                _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(
                    j,
                    MPI.rank,
                    _xsplits1,
                    _xsplits2,
                    _ysplits1,
                    _ysplits2,
                    reverse=False,
                )
                fnew[:, _ysplit1:_ysplit2, :] = _f
            fnew = np.array(fnew)
        else:
            _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(
                MPI.rank, i, _xsplits1, _xsplits2, _ysplits1, _ysplits2, reverse=False
            )
            MPI.send(f[_xsplit1:_xsplit2, :, :], to_rank=i, tag=i * MPI.size + MPI.rank)
        MPI.wait()
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
    f_fourier = redistribute_forward_2D(f_fourier, ngrid, MPI, iscomplex=True)
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
    _f_fourier = redistribute_backward_2D(_f_fourier, ngrid, MPI, iscomplex=True)
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
    f_fourier = redistribute_forward_2D(f_fourier, ngrid, MPI, iscomplex=False)
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
    _f_fourier = redistribute_backward_2D(_f_fourier, ngrid, MPI, iscomplex=False)
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
    f_fourier = redistribute_forward_2D(f_fourier, ngrid, MPI, iscomplex=False)
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
    _f_fourier = redistribute_backward_2D(_f_fourier, ngrid, MPI, iscomplex=False)
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
    f_fourier = redistribute_forward_3D(f_fourier, ngrid, MPI, iscomplex=True)
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
    _f_fourier = redistribute_backward_3D(_f_fourier, ngrid, MPI, iscomplex=True)
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
    f_fourier = redistribute_forward_3D(f_fourier, ngrid, MPI, iscomplex=False)
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
    _f_fourier = redistribute_backward_3D(_f_fourier, ngrid, MPI, iscomplex=False)
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
    f_fourier = redistribute_forward_3D(f_fourier, ngrid, MPI, iscomplex=False)
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
    _f_fourier = redistribute_backward_3D(_f_fourier, ngrid, MPI, iscomplex=False)
    # fft along y
    _f_fourier = slab_idst1D(_f_fourier, yboxsize, yngrid, axis=1, type=type)
    # fft along z
    f = slab_idst1D(_f_fourier, zboxsize, zngrid, axis=2, type=type)
    return f
