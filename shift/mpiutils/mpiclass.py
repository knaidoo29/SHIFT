import numpy as np

from typing import Any, List, Tuple, Optional, Union

from . import loops


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

    def wait(self):
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
        return loops.get_MPI_loop_size(loop_size, self.size)

    def mpi_ind2ind(self, mpi_ind: int) -> int:
        """
        Converts the MPI_ind of a distributed loop to the index of a full loop.

        Parameters
        ----------
        mpi_ind : int

        Yields
        ------
        Index of a full loop.
        """
        return loops.MPI_ind2ind(mpi_ind, self.rank, self.size, self.loop_size)

    def clean_loop(self):
        """
        Gets ride of loop_size definition.
        """
        self.loop_size = None

    def split(
        self, length: int, size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For splitting an array across nodes.

        Parameters
        ----------
        Length: int
            Length of the array to be split.
        size: int
            Size of the MPI.size (i.e. MPI task), if set then this will be used to be split
            for other reasons.

        Returns
        -------
        split1 : array_like
            The indices of the first element of the split array.
        split2 : array_like
            The indices of the last element of the split array.
        """
        if size is None:
            split_equal = length / self.size
        else:
            split_equal = length / size
        split_floor = np.floor(split_equal)
        split_remain = split_equal - split_floor
        if size is None:
            counts = split_floor * np.ones(self.size)
            counts[: int(np.round(split_remain * self.size, decimals=0))] += 1
        else:
            counts = split_floor * np.ones(size)
            counts[: int(np.round(split_remain * size, decimals=0))] += 1
        counts = counts.astype("int")
        if size is None:
            splits = np.zeros(self.size + 1, dtype="int")
        else:
            splits = np.zeros(size + 1, dtype="int")
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

    def check_partition(
        self, NDshape: List[int], NDshape_split: List[int]
    ) -> np.ndarray:
        """
        Returns a boolean array showing the axes that an array will be split along.

        Parameters
        ----------
        NDshape : list
            The shape of the N-dimensional array.
        NDshape_split : list
            The shape of the N-dimensional split array.

        Yields
        ------
        Boolean array showing whether array will not be split along a said axes.
        """
        return np.array([NDshape[i] == NDshape_split[i] for i in range(len(NDshape))])

    # TODO: Remove these two functions -- pretty certain these are defunct and no longer used elsewhere.

    # def create_split_ndarray(self, arrays_nd: np.ndarray, whichaxis: List[bool]) -> np.ndarray:
    #     """
    #     Splits a list of 1D arrays based on the data partitioning scheme. To be used with
    #     create_split_ngrid.

    #     Parameters
    #     ----------
    #     arrays_nd : array_like
    #         List of 1D arrays to be split.
    #     whichaxis : array_like
    #         Boolean array showing whether array will not be split along a said axes.

    #     Returns
    #     -------
    #     split_arrays : array_like
    #         Split list of 1D array.
    #     """
    #     split_arrays = []
    #     for i in range(0, len(arrays_nd)):
    #         _array = arrays_nd[i]
    #         if whichaxis[i] == False:
    #             _array = self.split_array(_array)
    #             split_arrays.append(_array)
    #         else:
    #             split_arrays.append(_array)
    #     return split_arrays

    # def create_split_ndgrid(self, arrays_nd: np.ndarray, whichaxis: List[bool]) -> np.ndarray:
    #     """
    #     Creates a partitioned gridded data set.

    #     Parameters
    #     ----------
    #     arrays_nd : array_like
    #         List of arrays to be split.
    #     whichaxis : array_like
    #         Boolean array showing whether array will not be split along a said axes.

    #     Returns
    #     -------
    #     split_grid : array_like
    #         N-dimensional split array.
    #     """
    #     split_arrays = self.create_split_ndarray(arrays_nd, whichaxis)
    #     split_grid = np.meshgrid(*split_arrays, indexing='ij')
    #     return split_grid

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

    def send(
        self, data: np.ndarray, to_rank: Optional[int] = None, tag: int = 11
    ) -> None:
        """
        Sends data from current core to other specified or all cores.

        Parameters
        ----------
        data : array
            Data to send.
        to_rank : int, optional
            Specify rank to send data to, or leave as None to send to all cores.
        tag : int, optional
            Sending tag to ensure the right data is being transfered.
        """
        if to_rank is not None:
            self.comm.send(data, dest=to_rank, tag=tag)
        else:
            for i in range(0, self.size):
                if i != self.rank:
                    self.comm.send(data, dest=i, tag=tag)

    def recv(self, from_rank: int, tag: int = 11) -> np.ndarray:
        """
        Receive data from another node.

        Parameters
        ----------
        from_rank : int
            Source of the data.
        tag : int
            Sending tag to ensure the right data is being transfered.

        Returns
        -------
        data : array
            Data received.
        """
        data = self.comm.recv(source=from_rank, tag=tag)
        return data

    def broadcast(self, data: Any) -> Any:
        """
        Broadcast data from rank=0 to all nodes.
        """
        if self.rank == 0:
            self.send(data, tag=11)
        else:
            data = self.recv(0, tag=11)
        self.wait()
        return data

    def send_up(self, data: Any) -> Any:  # pragma: no cover
        """
        Send data from each node to the node above.
        """

        datain = np.copy(data)

        if self.rank < self.size - 1:
            self.send(datain, to_rank=self.rank + 1, tag=10 + self.rank)
        if self.rank > 0:
            dataout = self.recv(self.rank - 1, tag=10 + self.rank - 1)

        self.wait()

        if self.rank == self.size - 1:
            self.send(datain, to_rank=0, tag=10 + self.size)
        if self.rank == 0:
            dataout = self.recv(self.size - 1, tag=10 + self.size)

        self.wait()

        return dataout

    def isend_up(self, data: Any) -> Any:
        """
        Send data from each node to the node above (rank+1, wrapping around).
        """
        datain = np.copy(data)

        # Non-blocking receive from below
        if self.rank > 0:
            req_recv = self.comm.irecv(source=self.rank - 1, tag=10 + self.rank - 1)
        else:
            req_recv = self.comm.irecv(source=self.size - 1, tag=10 + self.size)

        # Non-blocking send upward
        if self.rank < self.size - 1:
            req_send = self.comm.isend(
                obj=datain, dest=self.rank + 1, tag=10 + self.rank
            )
        else:
            req_send = self.comm.isend(obj=datain, dest=0, tag=10 + self.size)

        # Wait for receive and send to complete
        dataout = req_recv.wait()
        req_send.wait()
        return dataout

    def send_down(self, data: Any) -> Any:  # pragma: no cover
        """
        Send data from each node to the node below.
        """
        datain = np.copy(data)

        if self.rank > 0:
            self.send(datain, to_rank=self.rank - 1, tag=20 + self.rank)
        if self.rank < self.size - 1:
            dataout = self.recv(self.rank + 1, tag=20 + self.rank + 1)

        self.wait()

        if self.rank == self.size - 1:
            dataout = self.recv(0, tag=20 + self.size)
        if self.rank == 0:
            self.send(datain, to_rank=self.size - 1, tag=20 + self.size)

        self.wait()

        return dataout

    def isend_down(self, data: Any) -> Any:
        """
        Send data from each node to the node below (rank-1, wrapping around).
        """
        datain = np.copy(data)

        # Non-blocking receive from above
        if self.rank < self.size - 1:
            req_recv = self.comm.irecv(source=self.rank + 1, tag=20 + self.rank + 1)
        else:
            req_recv = self.comm.irecv(source=0, tag=20 + self.size)

        # Non-blocking send downward
        if self.rank > 0:
            req_send = self.comm.isend(
                obj=datain, dest=self.rank - 1, tag=20 + self.rank
            )
        else:
            req_send = self.comm.isend(
                obj=datain, dest=self.size - 1, tag=20 + self.size
            )

        # Wait for receive and send to complete
        dataout = req_recv.wait()
        req_send.wait()
        return dataout

    def collect(self, data: np.ndarray, outlist: bool = False) -> np.ndarray:
        """
        Collects a distributed data to the processor with rank=0.

        Parameters
        ----------
        data : array
            Distributed data set.
        outlist : bool, optional
            If outlist is False, we collect and concatenate, if True then
            we do not concatenate the list.
        """
        if np.isscalar(data):
            data = np.array([data])
        if self.rank == 0:
            datas = [data]
            for i in range(1, self.size):
                _data = self.recv(i, tag=10 + i)
                datas.append(_data)
            if outlist is False:
                data = np.concatenate(datas)
            else:
                data = datas
        else:
            self.send(data, to_rank=0, tag=10 + self.rank)
            data = None
        self.wait()
        return data

    def collect_noNone(self, data: np.ndarray) -> np.ndarray:
        """
        Same as collect function, but removes data=None to the combined data set.
        Parameters
        ----------
        data : array
            Distributed data set.
        """
        _datas = self.collect(data, outlist=True)
        if self.rank == 0:
            datas = []
            for i in range(0, len(_datas)):
                if _datas[i] is not None:
                    datas.append(_datas[i])
            datas = np.concatenate(datas)
        else:
            datas = None
        return datas

    def distribute(self, data: np.ndarray) -> np.ndarray:
        """
        Distribute and split data from rank 0.

        Parameters
        ----------
        data : array
            Full data set which will be split across the cores.
        """
        if self.rank == 0:
            split1, split2 = self.split(len(data))
            for i in range(1, len(split1)):
                self.send(data[split1[i] : split2[i]], to_rank=i, tag=10 + i)
            data = data[split1[0] : split2[0]]
        else:
            data = self.recv(0, tag=10 + self.rank)
        self.wait()
        return data

    def sum(
        self, data: Union[np.ndarray, int, float]
    ) -> Union[np.ndarray, int, float, None]:
        """
        Sums a distributed data set to the processor with rank=0.

        Parameters
        ----------
        data : array
            distributed data set.
        """
        if self.rank == 0:
            for i in range(1, self.size):
                _data = self.recv(i, tag=10 + i)
                data += _data
        else:
            self.send(data, to_rank=0, tag=10 + self.rank)
            data = None
        self.wait()
        return data

    def mean(self, data: np.ndarray) -> Union[float, int]:
        """
        Finds the mean of a distributed data set, which is broadcasted to all nodes.

        Parameters
        ----------
        data : array
            distributed data set.
        """
        total_data = self.sum(np.sum(data))
        self.wait()
        total_elem = self.sum(len(data.flatten()))
        self.wait()
        if self.rank == 0:
            mean = total_data / total_elem
            self.send(mean, tag=11)
        else:
            mean = self.recv(0, tag=11)
        self.wait()
        return mean

    def min(self, data: np.ndarray) -> Union[float, int]:
        """
        Finds the minimum of a distributed data set, which is broadcasted to all nodes.

        Parameters
        ----------
        data : array
            distributed data set.
        """
        mins = self.collect(np.min(data))
        if self.rank == 0:
            minval = np.min(mins)
            self.send(minval, tag=11)
        else:
            minval = self.recv(0, tag=11)
        self.wait()
        return minval

    def max(self, data: np.ndarray) -> Union[float, int]:
        """
        Finds the maximum of a distributed data set, which is broadcasted to all nodes.

        Parameters
        ----------
        data : array
            distributed data set.
        """
        maxs = self.collect(np.max(data))
        if self.rank == 0:
            maxval = np.max(maxs)
            self.send(maxval, tag=11)
        else:
            maxval = self.recv(0, tag=11)
        self.wait()
        return maxval

    def end(self):  # pragma: no cover
        """
        Ends MPI environment.
        """
        self.mpi.Finalize()
