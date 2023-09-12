import numpy as np
import scipy.fft as scfft


def slab_fft1D(f_real, boxsize, ngrid, axis=None):
    """Performs Forward FFT on input grid data.

    Parameters
    ----------
    f_real : ndarray
        Input grid data.
    boxsize : float
        Box size.
    axis : int, optional
        Axis to perform FFT.

    Returns
    -------
    f_fourier : ndarray
        Output Fourier grid data.
    """
    dx = boxsize / float(ngrid)
    if axis is None:
        f_fourier = scfft.fft(f_real)
    else:
        f_fourier = scfft.fft(f_real, axis=axis)
    f_fourier *= (dx/np.sqrt(2.*np.pi))
    return f_fourier


def slab_ifft1D(f_fourier, boxsize, ngrid, axis=None):
    """Performs backward fft of a Fourier grid.

    Parameters
    ----------
    f_fourier : ndarray
        Input Fourier grid data.
    boxsize : float
        Box size.
    ncpu : int, optional
        Number of cpus.
    axis : int, optional
        Axis to perform FFT.

    Returns
    -------
    f_real : ndarray
        Output grid data.
    """
    dx = boxsize / float(ngrid)
    if axis is None:
        f_real = scfft.ifft(f_fourier)
    else:
        f_real = scfft.ifft(f_fourier, axis=axis)
    f_real *= (np.sqrt(2.*np.pi)/dx)
    return f_real


def _get_splits_subset_2D(rank1, rank2, xsplits1, xsplits2, ysplits1, ysplits2,
    reverse=False):
    """Internal function for finding index splits across two axes based on input
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


def _get_splits_2D(MPI, xngrid, yngrid):
    """Finds the beginning and end index for splitting arrays in 2 dimensions.

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


def _get_splits_3D(MPI, xngrid, yngrid, zngrid):
    """Finds the beginning and end index for splitting arrays in 3 dimensions.

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


def _get_empty_split_array_2D(xsplits1, xsplits2, ysplits1, ysplits2, rank,
    axis=0, iscomplex=False):
    """Returns a normal or complex array with zeros.

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
        xsplit1, xsplit2 = xsplits1[rank], ysplits2[rank]
        ysplit1, ysplit2 = ysplits1[0], ysplits2[-1]
    else:
        xsplit1, xsplit2 = xsplits1[0], ysplits2[-1]
        ysplit1, ysplit2 = ysplits1[rank], ysplits2[rank]
    if iscomplex:
        return np.zeros((xsplit2-xsplit1, ysplit2-ysplit1)) + \
            1j*np.zeros((xsplit2-xsplit1, ysplit2-ysplit1))
    else:
        return np.zeros((xsplit2-xsplit1, ysplit2-ysplit1))


def _get_empty_split_array_3D(xsplits1, xsplits2, ysplits1, ysplits2, zsplits1,
    zsplits2, rank, axis=0, iscomplex=False):
    """Returns a normal or complex array with zeros.

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
        xsplit1, xsplit2 = xsplits1[rank], ysplits2[rank]
        ysplit1, ysplit2 = ysplits1[0], ysplits2[-1]
        zsplit1, zsplit2 = zsplits1[0], zsplits2[-1]
    else:
        xsplit1, xsplit2 = xsplits1[0], ysplits2[-1]
        ysplit1, ysplit2 = ysplits1[rank], ysplits2[rank]
        zsplit1, zsplit2 = zsplits1[0], zsplits2[-1]
    if iscomplex:
        return np.zeros((xsplit2-xsplit1, ysplit2-ysplit1, zsplit2-zsplit1)) + \
            1j*np.zeros((xsplit2-xsplit1, ysplit2-ysplit1, zsplit2-zsplit1))
    else:
        return np.zeros((xsplit2-xsplit1, ysplit2-ysplit1, zsplit2-zsplit1))


def redistribute_forward_2D(f, ngrid, MPI, iscomplex=False):
    """Redistributes a 2D array from the conventional axis split across x to a
    split across y.

    Parameters
    ----------
    f : 2darray
        Input data split across the x axis.
    ngrid : int
        Grid size along each axis.
    MPI : object
        MPIutils MPI object.
    iscomplex : bool, optional
        Is the input data complex.
    """
    _xsplits1, _xsplits2, _ysplits1, _ysplits2 = _get_splits_2D(MPI, ngrid, ngrid)
    for i in range(0, MPI.size):
        if i == MPI.rank:
            fnew = _get_empty_split_array_2D(_xsplits1, _xsplits2, _ysplits1,
                _ysplits2, MPI.rank, axis=1, iscomplex=iscomplex)
            for j in range(0, MPI.size):
                if j != MPI.rank:
                    _f = MPI.recv(j, tag=i*MPI.size+j)
                else:
                    _xsplit1, _xsplit2, _ysplit1, _ysplit2 = \
                        _get_splits_subset_2D(MPI.rank, i, _xsplits1, _xsplits2,
                            _ysplits1, _ysplits2, reverse=True)
                    _f = f[:,_ysplit1:_ysplit2]
                _xsplit1, _xsplit2, _ysplit1, _ysplit2 = \
                    _get_splits_subset_2D(j, MPI.rank, _xsplits1, _xsplits2,
                        _ysplits1, _ysplits2, reverse=True)
                fnew[_xsplit1:_xsplit2,:] = _f
            fnew = np.array(fnew)
        else:
            _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(MPI.rank,
                i, _xsplits1, _xsplits2, _ysplits1, _ysplits2, reverse=True)
            MPI.send(f[:,_ysplit1:_ysplit2], to_rank=i, tag=i*MPI.size+MPI.rank)
        MPI.wait()
    return fnew


def redistribute_forward_3D(f, ngrid, MPI, iscomplex=False):
    """Redistributes a 3D array from the conventional axis split across x to a
    split across y.

    Parameters
    ----------
    f : 3darray
        Input data split across the x axis.
    ngrid : int
        Grid size along each axis.
    MPI : object
        MPIutils MPI object.
    iscomplex : bool, optional
        Is the input data complex.
    """
    _xsplits1, _xsplits2, _ysplits1, _ysplits2, _zsplits1, _zsplits2 = \
        _get_splits_3D(MPI, ngrid, ngrid, ngrid)
    for i in range(0, MPI.size):
        if i == MPI.rank:
            fnew = _get_empty_split_array_3D(_xsplits1, _xsplits2, _ysplits1,
                _ysplits2, _zsplits1, _zsplits2, MPI.rank, axis=1,
                iscomplex=iscomplex)
            for j in range(0, MPI.size):
                if j != MPI.rank:
                    _f = MPI.recv(j, tag=i*MPI.size+j)
                else:
                    _xsplit1, _xsplit2, _ysplit1, _ysplit2 = \
                        _get_splits_subset_2D(MPI.rank, i, _xsplits1, _xsplits2,
                            _ysplits1, _ysplits2, reverse=True)
                    _f = f[:,_ysplit1:_ysplit2,:]
                _xsplit1, _xsplit2, _ysplit1, _ysplit2 = \
                    _get_splits_subset_2D(j, MPI.rank, _xsplits1, _xsplits2,
                        _ysplits1, _ysplits2, reverse=True)
                fnew[_xsplit1:_xsplit2,:,:] = _f
            fnew = np.array(fnew)
        else:
            _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(MPI.rank,
                i, _xsplits1, _xsplits2, _ysplits1, _ysplits2, reverse=True)
            MPI.send(f[:,_ysplit1:_ysplit2,:], to_rank=i, tag=i*MPI.size+MPI.rank)
        MPI.wait()
    return fnew


def redistribute_backward_2D(f, ngrid, MPI, iscomplex=False):
    """Redistributes a 2D array backwards a split across y to a split across x.

    Parameters
    ----------
    f : 2darray
        Input data split across the x axis.
    ngrid : int
        Grid size along each axis.
    MPI : object
        MPIutils MPI object.
    iscomplex : bool, optional
        Is the input data complex.
    """
    _xsplits1, _xsplits2, _ysplits1, _ysplits2 = _get_splits_2D(MPI, ngrid, ngrid)
    for i in range(0, MPI.size):
        if i == MPI.rank:
            fnew = _get_empty_split_array_2D(_xsplits1, _xsplits2, _ysplits1,
                _ysplits2, MPI.rank, axis=0,
                iscomplex=iscomplex)
            for j in range(0, MPI.size):
                if j != MPI.rank:
                    _f = MPI.recv(j, tag=i*MPI.size+j)
                else:
                    _xsplit1, _xsplit2, _ysplit1, _ysplit2 = \
                        _get_splits_subset_2D(MPI.rank, i, _xsplits1, _xsplits2,
                            _ysplits1, _ysplits2, reverse=False)
                    _f = f[_xsplit1:_xsplit2,:]
                _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(j,
                    MPI.rank, _xsplits1, _xsplits2, _ysplits1, _ysplits2, reverse=False)
                fnew[:,_ysplit1:_ysplit2] = _f
            fnew = np.array(fnew)
        else:
            _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(MPI.rank,
                i, _xsplits1, _xsplits2, _ysplits1, _ysplits2, reverse=False)
            MPI.send(f[_xsplit1:_xsplit2,:], to_rank=i, tag=i*MPI.size+MPI.rank)
        MPI.wait()
    return fnew


def redistribute_backward_3D(f, ngrid, MPI, iscomplex=False):
    """Redistributes a 3D array backwards a split across y to a split across x.

    Parameters
    ----------
    f : 3darray
        Input data split across the x axis.
    ngrid : int
        Grid size along each axis.
    MPI : object
        MPIutils MPI object.
    iscomplex : bool, optional
        Is the input data complex.
    """
    _xsplits1, _xsplits2, _ysplits1, _ysplits2, _zsplits1, _zsplits2 = \
        _get_splits_3D(MPI, ngrid, ngrid, ngrid)
    for i in range(0, MPI.size):
        if i == MPI.rank:
            fnew = _get_empty_split_array_3D(_xsplits1, _xsplits2, _ysplits1,
                _ysplits2, _zsplits1, _zsplits2, MPI.rank, axis=0,
                iscomplex=iscomplex)
            for j in range(0, MPI.size):
                if j != MPI.rank:
                    _f = MPI.recv(j, tag=i*MPI.size+j)
                else:
                    _xsplit1, _xsplit2, _ysplit1, _ysplit2 = \
                        _get_splits_subset_2D(MPI.rank, i, _xsplits1, _xsplits2,
                            _ysplits1, _ysplits2, reverse=False)
                    _f = f[_xsplit1:_xsplit2,:,:]
                _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(j,
                    MPI.rank, _xsplits1, _xsplits2, _ysplits1, _ysplits2, reverse=False)
                fnew[:,_ysplit1:_ysplit2,:] = _f
            fnew = np.array(fnew)
        else:
            _xsplit1, _xsplit2, _ysplit1, _ysplit2 = _get_splits_subset_2D(MPI.rank,
                i, _xsplits1, _xsplits2, _ysplits1, _ysplits2, reverse=False)
            MPI.send(f[_xsplit1:_xsplit2,:,:], to_rank=i, tag=i*MPI.size+MPI.rank)
        MPI.wait()
    return fnew


def mpi_fft2D(f_real, boxsize, ngrid, MPI):
    """Performs MPI Forward FFT on input grid data.

    Parameters
    ----------
    f_real : ndarray
        Input grid data, assumed to be real.
    boxsize : float
        Box size.
    ngrid : int
        Size of the grid along one axis.
    MPI : obj
        MPIutils MPI object.

    Returns
    -------
    f_fourier : ndarray
        Output Fourier grid data.
    """
    f_fourier = slab_fft1D(f_real, boxsize, ngrid, axis=1)
    f_fourier = redistribute_forward_2D(f_fourier, ngrid, MPI, iscomplex=True)
    f_fourier = slab_fft1D(f_fourier, boxsize, ngrid, axis=0)
    return f_fourier


def mpi_ifft2D(f_fourier, boxsize, ngrid, MPI):
    """Performs MPI Forward FFT on input grid data.

    Parameters
    ----------
    f_fourier : ndarray
        Output Fourier grid data.
    boxsize : float
        Box size.
    ngrid : int
        Size of the grid along one axis.
    MPI : obj
        MPIutils MPI object.

    Returns
    -------
    f : ndarray
        Input grid data.
    """
    _f_fourier = slab_ifft1D(f_fourier, boxsize, ngrid, axis=0)
    _f_fourier = redistribute_backward_2D(_f_fourier, ngrid, MPI, iscomplex=True)
    f = slab_ifft1D(_f_fourier, boxsize, ngrid, axis=1)
    return f.real


def mpi_fft3D(f_real, boxsize, ngrid, MPI):
    """Performs MPI Forward FFT on input grid data.

    Parameters
    ----------
    f_real : ndarray
        Input grid data, assumed to be real.
    boxsize : float
        Box size.
    ngrid : int
        Size of the grid along one axis.
    MPI : obj
        MPIutils MPI object.

    Returns
    -------
    f_fourier : ndarray
        Output Fourier grid data.
    """
    f_fourier = slab_fft1D(f_real, boxsize, ngrid, axis=1)
    f_fourier = slab_fft1D(f_fourier, boxsize, ngrid, axis=2)
    f_fourier = redistribute_forward_3D(f_fourier, ngrid, MPI, iscomplex=True)
    f_fourier = slab_fft1D(f_fourier, boxsize, ngrid, axis=0)
    return f_fourier


def mpi_ifft3D(f_fourier, boxsize, ngrid, MPI):
    """Performs MPI Forward FFT on input grid data.

    Parameters
    ----------
    f_fourier : ndarray
        Output Fourier grid data.
    boxsize : float
        Box size.
    ngrid : int
        Size of the grid along one axis.
    MPI : obj
        MPIutils MPI object.

    Returns
    -------
    f : ndarray
        Input grid data.
    """
    _f_fourier = slab_ifft1D(f_fourier, boxsize, ngrid, axis=0)
    _f_fourier = redistribute_backward_3D(_f_fourier, ngrid, MPI, iscomplex=True)
    _f_fourier = slab_ifft1D(_f_fourier, boxsize, ngrid, axis=1)
    f = slab_ifft1D(_f_fourier, boxsize, ngrid, axis=2)
    return f.real
