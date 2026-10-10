import numpy as np

from typing import Any, List, Tuple, Optional, Union


def get_MPI_loop_size(loop_size: int, MPI_size: int) -> int:
    """
    Converts a loop size to the MPI loop size which distributes the loop across
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
    MPI_loop_size = int(np.floor(loop_size / MPI_size))
    if MPI_loop_size * MPI_size < loop_size:
        MPI_loop_size += 1
    return MPI_loop_size


class MPI:


    def __init__(self):
        """
        Initialises MPI.
        """
        from mpi4py import MPI as mpi

        self.mpi = mpi
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.loop_size = None
        self.mpi_info = "Proc " + str(self.rank + 1) + " of " + str(self.size)


    def Barrier(self):
        """
        Tells all jobs to wait -- to ensure jobs are synchronised.
        """
        self.comm.Barrier()


    def set_loop(self, loop_size: int) -> int:
        """
        Sets the size of a distributed loop.

        Parameters
        ----------
        loop_size : int

        Yields
        ------
        Size of the MPI_loop.
        """
        self.loop_size = loop_size
        return get_MPI_loop_size(loop_size, self.size)
    

    def clean_loop(self):
        """
        Gets ride of loop_size definition.
        """
        self.loop_size = None


    def split(
        self,
        length: int,
        size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the beginning and end indices for splitting an array across ranks.

        The array is divided as evenly as possible, with any remainder assigned
        to the lower-ranked partitions.

        Parameters
        ----------
        length : int
            Length of the array to split.
        size : int, optional
            Number of partitions. If not provided, ``self.size`` is used.

        Returns
        -------
        split1 : np.ndarray
            Starting index of each partition.
        split2 : np.ndarray
            Ending index of each partition.
        """
        if length < 0:
            raise ValueError("length must be non-negative.")

        nparts = self.size if size is None else size

        if nparts <= 0:
            raise ValueError("size must be greater than zero.")

        base, remainder = divmod(length, nparts)

        counts = np.full(nparts, base, dtype=int)
        counts[:remainder] += 1

        splits = np.empty(nparts + 1, dtype=int)
        splits[0] = 0
        splits[1:] = np.cumsum(counts)

        split1 = splits[:-1]
        split2 = splits[1:]

        return split1, split2

    
    def split_array(self, array: np.ndarray) -> np.ndarray:
        """
        Returns the values of the split array.

        Parameters
        ----------
        array : array_like
            Array to be split.

        Yields
        ------
        Split array.
        """
        split1, split2 = self.split(len(array))
        return array[split1[self.rank] : split2[self.rank]]


    def mpi_print(self, *value: Any) -> None:
        """
        Python print function using flush=True so print statements are outputed
        immediately in an MPI setting.
        """
        print(*value, flush=True)


    def mpi_print_zero(self, *value: Any) -> None:
        """
        Prints only at node rank = 0.
        """
        if self.rank == 0:
            self.mpi_print(*value)
    

    def Bcast(self, data: np.ndarray, root: int = 0) -> None:
        """
        Broadcast a NumPy array from one MPI rank to all ranks.

        This uses the buffer-based ``mpi4py`` broadcast operation and therefore
        communicates the array directly without Python object serialization.

        Parameters
        ----------
        data : np.ndarray
            Array buffer to broadcast. On the root rank, this contains the data
            to be sent. On all other ranks, this must be an allocated array with
            the same shape and dtype as the root array. The receive buffers are
            overwritten in place.
        root : int, optional
            Rank from which the data are broadcast. Default is 0.

        Returns
        -------
        None
            The input array is modified in place on non-root ranks.

        Notes
        -----
        All MPI ranks in the communicator must call this function collectively
        with the same value of ``root``.

        The shape and dtype of ``data`` must be compatible across all ranks.
        """
        self.comm.Bcast(data, root=root)


    def bcast(self, data, root=0):
        """
        Broadcast a Python object from root to all MPI ranks.
        """
        return self.comm.bcast(data, root=root)


    def Send_up(self, data: np.ndarray, tag: int = 10) -> np.ndarray:
        """
        Send an array to the next MPI rank and receive an array from the
        previous MPI rank.

        The rank ordering is periodic, such that the final rank sends to rank 0
        and rank 0 receives from the final rank.

        Parameters
        ----------
        data : np.ndarray
            Array to send to the next MPI rank.
        tag : int, optional
            MPI message tag used for the exchange. Default is 10.

        Returns
        -------
        dataout : np.ndarray
            Array received from the previous MPI rank.
        """
        source = (self.rank - 1) % self.size
        dest = (self.rank + 1) % self.size

        datain = np.ascontiguousarray(data)
        dataout = np.empty_like(datain, dtype=data.dtype)

        self.comm.Sendrecv(
            sendbuf=[datain.view(np.uint8), self.mpi.BYTE],
            dest=dest,
            sendtag=tag,
            recvbuf=[dataout.view(np.uint8), self.mpi.BYTE],
            source=source,
            recvtag=tag,
        )

        return dataout  


    def Send_up_variable(
        self,
        data: np.ndarray,
        tag: int = 10,
    ) -> np.ndarray:
        """
        Send a variable-length array to the next MPI rank and receive a
        variable-length array from the previous MPI rank.

        The rank ordering is periodic, such that the final rank sends to rank 0
        and rank 0 receives from the final rank.

        The array may vary in length along its first axis. Any trailing
        dimensions and the dtype must be compatible between neighbouring ranks.

        Parameters
        ----------
        data : np.ndarray
            Array to send to the next MPI rank.
        tag : int, optional
            Base MPI message tag used for the exchange. The size exchange uses
            ``tag`` and the data exchange uses ``tag + 1``. Default is 10.

        Returns
        -------
        dataout : np.ndarray
            Array received from the previous MPI rank.
        """
        source = (self.rank - 1) % self.size
        dest = (self.rank + 1) % self.size

        datain = np.ascontiguousarray(data)

        # Exchange the length of the first axis.
        sendcount = np.array([datain.shape[0]], dtype=np.int64)
        recvcount = np.empty(1, dtype=np.int64)

        self.comm.Sendrecv(
            sendbuf=sendcount,
            dest=dest,
            sendtag=tag,
            recvbuf=recvcount,
            source=source,
            recvtag=tag,
        )

        # Allocate the receive buffer with matching trailing dimensions.
        dataout = np.empty(
            (recvcount[0], *datain.shape[1:]),
            dtype=datain.dtype,
        )

        
        # Exchange the array data.
        self.comm.Sendrecv(
            sendbuf=[datain.view(np.uint8), self.mpi.BYTE],
            dest=dest,
            sendtag=tag + 1,
            recvbuf=[dataout.view(np.uint8), self.mpi.BYTE],
            source=source,
            recvtag=tag + 1,
        )

        return dataout


    def Send_down(
        self,
        data: np.ndarray,
        tag: int = 20,
    ) -> np.ndarray:
        """
        Send an array to the previous MPI rank and receive an array from the
        next MPI rank.

        The rank ordering is periodic, such that rank 0 sends to the final rank
        and the final rank receives from rank 0.

        Parameters
        ----------
        data : np.ndarray
            Array to send to the previous MPI rank.
        tag : int, optional
            MPI message tag used for the exchange. Default is 20.

        Returns
        -------
        dataout : np.ndarray
            Array received from the next MPI rank.
        """
        dest = (self.rank - 1) % self.size
        source = (self.rank + 1) % self.size

        datain = np.ascontiguousarray(data)
        dataout = np.empty_like(datain, dtype=data.dtype)
        
        self.comm.Sendrecv(
            sendbuf=[datain.view(np.uint8), self.mpi.BYTE],
            dest=dest,
            sendtag=tag,
            recvbuf=[dataout.view(np.uint8), self.mpi.BYTE],
            source=source,
            recvtag=tag,
        )

        return dataout


    def Send_down_variable(
        self,
        data: np.ndarray,
        tag: int = 20,
    ) -> np.ndarray:
        """
        Send a variable-length array to the previous MPI rank and receive a
        variable-length array from the next MPI rank.

        The rank ordering is periodic, such that rank 0 sends to the final rank
        and the final rank receives from rank 0.

        The array is assumed to vary only along the first axis. Any trailing
        dimensions must be the same on all ranks.

        Parameters
        ----------
        data : np.ndarray
            Array to send to the previous MPI rank.
        tag : int, optional
            Base MPI message tag used for the exchange. The size exchange uses
            ``tag`` and the data exchange uses ``tag + 1``. Default is 20.

        Returns
        -------
        dataout : np.ndarray
            Array received from the next MPI rank.

        Notes
        -----
        This function first exchanges the length of the first axis, then
        allocates the required receive buffer, and finally exchanges the data.

        All ranks must call this function collectively with compatible trailing
        dimensions and dtype.
        """
        dest = (self.rank - 1) % self.size
        source = (self.rank + 1) % self.size

        datain = np.ascontiguousarray(data)

        # Exchange the length of the first axis.
        sendcount = np.array([datain.shape[0]], dtype=np.int64)
        recvcount = np.empty(1, dtype=np.int64)

        self.comm.Sendrecv(
            sendbuf=sendcount,
            dest=dest,
            sendtag=tag,
            recvbuf=recvcount,
            source=source,
            recvtag=tag,
        )

        # Allocate the receive buffer with matching trailing dimensions.
        dataout = np.empty(
            (recvcount[0], *datain.shape[1:]),
            dtype=datain.dtype,
        )

        # Exchange the array data.
        self.comm.Sendrecv(
            sendbuf=[datain.view(np.uint8), self.mpi.BYTE],
            dest=dest,
            sendtag=tag + 1,
            recvbuf=[dataout.view(np.uint8), self.mpi.BYTE],
            source=source,
            recvtag=tag + 1,
        )

        return dataout


    def gather(self, data, root: int = 0):
        """
        Gather Python objects from all MPI ranks onto one rank.

        Unlike ``Gather``, this operation preserves each rank's contribution
        as a separate Python object rather than concatenating numerical arrays.

        Parameters
        ----------
        data : Any
            Python object contributed by the local MPI rank.
        root : int, optional
            Rank on which the gathered objects are returned. Default is 0.

        Returns
        -------
        dataout : list or None
            List containing one object from each MPI rank on the root rank.
            Returns None on all other ranks.

        Notes
        -----
        This function uses the object-based ``mpi4py`` ``gather`` collective
        and therefore uses Python serialization.
        """
        return self.comm.gather(data, root=root)

    
    def Gather(
        self,
        data: np.ndarray,
        root: int = 0,
    ) -> np.ndarray | None:
        """
        Gather distributed NumPy arrays onto one MPI rank.

        Uses ``Gather`` when all ranks contribute arrays of equal length and
        ``Gatherv`` when the lengths differ.

        Parameters
        ----------
        data : np.ndarray
            One-dimensional array held by the local MPI rank.
        root : int, optional
            Rank on which the collected array is returned. Default is 0.

        Returns
        -------
        dataout : np.ndarray or None
            Concatenated array on the root rank. None on all other ranks.
        """
        data = np.ascontiguousarray(data)

        # Determine contribution size from each rank.
        nlocal = np.array([data.size], dtype=np.int64)
        counts = np.empty(self.size, dtype=np.int64)

        self.comm.Allgather(nlocal, counts)

        equal_size = np.all(counts == counts[0])

        if equal_size:

            if self.rank == root:
                dataout = np.empty(counts[0] * self.size, dtype=data.dtype)
            else:
                dataout = None

            self.comm.Gather(data, dataout, root=root)

        else:

            displs = np.zeros(self.size, dtype=np.int64)
            displs[1:] = np.cumsum(counts[:-1])

            if self.rank == root:
                dataout = np.empty(np.sum(counts), dtype=data.dtype,)
            else:
                dataout = None

            self.comm.Gatherv(data, [dataout, counts, displs], root=root)

        return dataout


    def Scatter(
        self,
        data: np.ndarray | None,
        dtype: np.dtype,
        root: int = 0,
    ) -> np.ndarray:
        """
        Split a one-dimensional NumPy array across MPI ranks.

        Uses ``Scatter`` when all ranks receive equal-sized chunks and
        ``Scatterv`` otherwise.

        Parameters
        ----------
        data : np.ndarray or None
            Full array on the root rank. May be None on all other ranks.
        dtype : np.dtype
            Data type of the distributed array.
        root : int, optional
            Rank containing the full input array. Default is 0.

        Returns
        -------
        data_local : np.ndarray
            Portion of the array assigned to the local MPI rank.
        """
        dtype = np.dtype(dtype)

        # Determine the global array length.
        n = np.empty(1, dtype=np.int64)

        if self.rank == root:
            if data is None:
                raise ValueError("data cannot be None on the root rank.")

            data = np.ascontiguousarray(data)

            if data.ndim != 1:
                raise ValueError("Scatter currently supports one-dimensional arrays.")

            if data.dtype != dtype:
                raise TypeError(
                    f"Input dtype {data.dtype} does not match requested dtype {dtype}."
                )

            n[0] = data.size

        self.comm.Bcast(n, root=root)

        split1, split2 = self.split(int(n[0]))
        counts = split2 - split1

        data_local = np.empty(
            counts[self.rank],
            dtype=dtype,
        )

        sendbuf = data if self.rank == root else None

        if np.all(counts == counts[0]):
            self.comm.Scatter(
                sendbuf,
                data_local,
                root=root,
            )
        else:
            if self.rank == root:
                sendbuf = [data, counts, split1]

            self.comm.Scatterv(
                sendbuf,
                data_local,
                root=root,
            )

        return data_local


    def Alltoall(
        self,
        sendbuf: np.ndarray,
        recvbuf: np.ndarray,
    ) -> None:
        """
        Perform an all-to-all exchange between MPI ranks.

        Each MPI rank sends an equal number of elements to every other rank and
        receives an equal number of elements from every other rank. The send and
        receive buffers must therefore be compatible with an equal partition
        across all ranks.

        Parameters
        ----------
        sendbuf : np.ndarray
            Contiguous buffer containing the data to send. The buffer is divided
            equally between destination MPI ranks.
        recvbuf : np.ndarray
            Preallocated buffer into which the received data are written. The
            buffer must have a compatible shape and dtype with ``sendbuf``.

        Returns
        -------
        None
            The received data are written into ``recvbuf`` in place.

        Notes
        -----
        This function uses the buffer-based ``mpi4py`` ``Alltoall`` collective
        and does not perform Python object serialization.

        All MPI ranks in the communicator must call this function collectively.
        """
        self.comm.Alltoall(sendbuf, recvbuf)


    def Alltoallv(
        self,
        sendbuf: np.ndarray,
        sendcounts: np.ndarray,
        senddispls: np.ndarray,
        recvbuf: np.ndarray,
        recvcounts: np.ndarray,
        recvdispls: np.ndarray,
    ) -> None:
        """
        Perform a variable-size all-to-all exchange between MPI ranks.

        Each MPI rank may send a different number of elements to every other
        rank and receive a different number of elements from every other rank.

        Parameters
        ----------
        sendbuf : np.ndarray
            Contiguous buffer containing all data to send.
        sendcounts : np.ndarray
            Number of elements in ``sendbuf`` to send to each destination rank.
            Must have length equal to the number of MPI ranks.
        senddispls : np.ndarray
            Starting displacement in ``sendbuf`` for the data sent to each
            destination rank. Must have length equal to the number of MPI ranks.
        recvbuf : np.ndarray
            Preallocated buffer into which the received data are written.
        recvcounts : np.ndarray
            Number of elements expected from each source rank. Must have length
            equal to the number of MPI ranks.
        recvdispls : np.ndarray
            Starting displacement in ``recvbuf`` for the data received from each
            source rank. Must have length equal to the number of MPI ranks.

        Returns
        -------
        None
            The received data are written into ``recvbuf`` in place.

        Notes
        -----
        Counts and displacements are specified in units of array elements, not
        bytes.

        This function uses the buffer-based ``mpi4py`` ``Alltoallv`` collective
        and does not perform Python object serialization.

        All MPI ranks in the communicator must call this function collectively.
        """
        self.comm.Alltoallv(
            [sendbuf, sendcounts, senddispls],
            [recvbuf, recvcounts, recvdispls],
        )

    def Alltoall_arrays(
        self,
        data: np.ndarray,
        ranks: np.ndarray,
    ) -> np.ndarray:
        """
        Redistribute entries of an array to specified MPI ranks.

        The first axis of ``data`` identifies independently distributed entries.
        Entries are grouped by destination rank and exchanged using
        ``Alltoallv``.

        Parameters
        ----------
        data : np.ndarray
            Array whose first axis identifies independently distributed entries.
            All ranks must use compatible trailing dimensions and dtype.
        ranks : np.ndarray
            Integer destination MPI rank for each entry along the first axis.

        Returns
        -------
        dataout : np.ndarray
            Entries assigned to the local MPI rank.

        Notes
        -----
        Entries sent from a given source rank preserve their relative ordering.
        Received entries are grouped in source-rank order.
        """
        data = np.asarray(data)
        ranks = np.asarray(ranks)

        if data.ndim == 0:
            raise ValueError("data must have at least one dimension.")

        if ranks.ndim != 1:
            raise ValueError("ranks must be a one-dimensional array.")

        if data.shape[0] != ranks.size:
            raise ValueError(
                "The number of destination ranks must match data.shape[0]."
            )

        if not np.issubdtype(ranks.dtype, np.integer):
            raise TypeError("ranks must contain integer MPI ranks.")

        if np.any((ranks < 0) | (ranks >= self.size)):
            raise ValueError("Destination MPI rank is out of range.")

        # Group entries by destination while preserving relative ordering
        # within each destination.
        order = np.argsort(ranks, kind="stable")
        senddata = np.ascontiguousarray(data[order])

        # Number of rows being sent to each rank.
        sendcounts_rows = np.bincount(
            ranks,
            minlength=self.size,
        ).astype(np.int64)

        recvcounts_rows = np.empty_like(sendcounts_rows)

        self.comm.Alltoall(
            sendcounts_rows,
            recvcounts_rows,
        )

        # Number of scalar elements in each entry.
        row_size = int(np.prod(data.shape[1:], dtype=np.int64))

        sendcounts = sendcounts_rows * row_size
        recvcounts = recvcounts_rows * row_size

        senddispls = np.zeros(self.size, dtype=np.int64)
        recvdispls = np.zeros(self.size, dtype=np.int64)

        senddispls[1:] = np.cumsum(sendcounts[:-1])
        recvdispls[1:] = np.cumsum(recvcounts[:-1])

        sendbuf = senddata.reshape(-1)

        recvbuf = np.empty(
            int(recvcounts.sum()),
            dtype=data.dtype,
        )

        self.comm.Alltoallv(
            [sendbuf, sendcounts, senddispls],
            [recvbuf, recvcounts, recvdispls],
        )

        return recvbuf.reshape(
            (-1, *data.shape[1:])
        )
    

    def Sum(
        self,
        data: Union[np.ndarray, int, float],
        root: int = 0,
    ) -> Union[np.ndarray, int, float, None]:
        """
        Sum values across all MPI ranks onto a single rank.

        Parameters
        ----------
        data : np.ndarray, int or float
            Local value or array to include in the global sum.
        root : int, optional
            MPI rank on which the summed result is returned. Default is 0.

        Returns
        -------
        result : np.ndarray, int, float or None
            Sum of the input values across all MPI ranks on the root rank.
            Returns None on all other ranks.
        """
        sendbuf = np.asarray(data)

        if self.rank == root:
            recvbuf = np.empty_like(sendbuf)
        else:
            recvbuf = None

        self.comm.Reduce(
            sendbuf,
            recvbuf,
            op=self.mpi.SUM,
            root=root,
        )

        if self.rank != root:
            return None

        if recvbuf.ndim == 0:
            return recvbuf.item()

        return recvbuf


    def Mean(self, data: np.ndarray):
        """
        Compute the mean of a distributed array across all MPI ranks.

        The calculation preserves the natural precision of the input array.
        Integer arrays are promoted to double precision, consistent with the
        usual behaviour of NumPy mean operations.

        Parameters
        ----------
        data : np.ndarray
            Local portion of the distributed array.

        Returns
        -------
        mean : scalar
            Global mean, returned on every MPI rank.
        """
        data = np.asarray(data)

        # Match the natural precision used for the mean.
        if np.issubdtype(data.dtype, np.integer):
            dtype = np.float64
        elif data.dtype == np.float16:
            dtype = np.float32
        else:
            dtype = data.dtype

        local_sum = np.array(
            [np.sum(data, dtype=dtype)],
            dtype=dtype,
        )
        global_sum = np.empty_like(local_sum)

        local_count = np.array(
            [data.size],
            dtype=np.int64,
        )
        global_count = np.empty_like(local_count)

        self.comm.Allreduce(
            local_sum,
            global_sum,
            op=self.mpi.SUM,
        )

        self.comm.Allreduce(
            local_count,
            global_count,
            op=self.mpi.SUM,
        )

        if global_count[0] == 0:
            raise ValueError(
                "Cannot compute the mean of an empty distributed array."
            )

        return dtype.type(global_sum[0] / global_count[0])


    def Min(self, data: np.ndarray):
        """
        Find the global minimum across all MPI ranks.

        Empty local arrays are ignored when determining the global minimum.

        Parameters
        ----------
        data : np.ndarray
            Local portion of the distributed array.

        Returns
        -------
        minval : scalar
            Global minimum, returned on every MPI rank.
        """
        data = np.asarray(data)

        if np.issubdtype(data.dtype, np.floating):
            local_min = np.inf if data.size == 0 else np.min(data)

        elif np.issubdtype(data.dtype, np.integer):
            local_min = (
                np.iinfo(data.dtype).max
                if data.size == 0
                else np.min(data)
            )

        elif np.issubdtype(data.dtype, np.bool_):
            local_min = True if data.size == 0 else np.min(data)

        else:
            raise TypeError(
                f"Min does not support dtype {data.dtype}."
            )

        sendbuf = np.array([local_min], dtype=data.dtype)
        recvbuf = np.empty_like(sendbuf)

        self.comm.Allreduce(
            sendbuf,
            recvbuf,
            op=self.mpi.MIN,
        )

        return recvbuf[0]


    def Max(self, data: np.ndarray):
        """
        Find the global maximum across all MPI ranks.

        Empty local arrays are ignored when determining the global maximum.

        Parameters
        ----------
        data : np.ndarray
            Local portion of the distributed array.

        Returns
        -------
        maxval : scalar
            Global maximum, returned on every MPI rank.
        """
        data = np.asarray(data)

        if np.issubdtype(data.dtype, np.floating):
            local_max = -np.inf if data.size == 0 else np.max(data)

        elif np.issubdtype(data.dtype, np.integer):
            local_max = (
                np.iinfo(data.dtype).min
                if data.size == 0
                else np.max(data)
            )

        elif np.issubdtype(data.dtype, np.bool_):
            local_max = False if data.size == 0 else np.max(data)

        else:
            raise TypeError(
                f"Max does not support dtype {data.dtype}."
            )

        sendbuf = np.array([local_max], dtype=data.dtype)
        recvbuf = np.empty_like(sendbuf)

        self.comm.Allreduce(
            sendbuf,
            recvbuf,
            op=self.mpi.MAX,
        )

        return recvbuf[0]
    

    def end(self): # pragma: no cover
        """
        Ends MPI environment.
        """
        self.mpi.Finalize()
