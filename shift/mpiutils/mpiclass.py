import numpy as np

from . import loops


class MPI:

    def __init__(self):
        """Initialises MPI."""
        from mpi4py import MPI as mpi
        self.mpi = mpi
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.loop_size = None
        self.mpi_info = 'Proc ' + str(self.rank+1)+' of ' + str(self.size)

    
    def wait(self):
        """Makes all jobs wait so they are synchronised."""
        self.comm.Barrier()


    def set_loop(self, loop_size):
        """Sets the size of a distributed loop."""
        self.loop_size = loop_size
        return loops.get_MPI_loop_size(loop_size, self.size)


    def mpi_ind2ind(self, mpi_ind):
        """Converts the MPI_ind of a distributed loop to the index of a full loop."""
        return loops.MPI_ind2ind(mpi_ind, self.rank, self.size, self.loop_size)


    def clean_loop(self):
        """Gets ride of loop_size definition."""
        self.loop_size = None


    def split(self, length, size=None):
        """For splitting an array across nodes."""
        if size is None:
            split_equal = length/self.size
        else:
            split_equal = length/size
        split_floor = np.floor(split_equal)
        split_remain = split_equal - split_floor
        if size is None:
            counts = split_floor*np.ones(self.size)
            counts[:int(np.round(split_remain*self.size, decimals=0))] += 1
        else:
            counts = split_floor*np.ones(size)
            counts[:int(np.round(split_remain*size, decimals=0))] += 1
        counts = counts.astype('int')
        if size is None:
            splits = np.zeros(self.size+1, dtype='int')
        else:
            splits = np.zeros(size+1, dtype='int')
        splits[1:] = np.cumsum(counts)
        split1 = splits[:-1]
        split2 = splits[1:]
        return split1, split2


    def split_array(self, array):
        """Returns the values of the split array."""
        split1, split2 = self.split(len(array))
        return array[split1[self.rank]:split2[self.rank]]


    def check_partition(self, NDshape, NDshape_split):
        """Returns bool array showing which axes the array is being split."""
        return np.array([NDshape[i] == NDshape_split[i] for i in range(len(NDshape))])


    def create_split_ndarray(self, arrays_nd, whichaxis):
        """Split a list of arrays based on the data partitioning scheme."""
        split_arrays = []
        for i in range(0, len(arrays_nd)):
            _array = arrays_nd[i]
            if not whichaxis[i]:
                _array = self.split_array(_array)
                split_arrays.append(_array)
            else:
                split_arrays.append(_array)
        return split_arrays


    def create_split_ndgrid(self, arrays_nd, whichaxis):
        """Create a partitioned gridded data set."""
        split_arrays = self.create_split_ndarray(arrays_nd, whichaxis)
        split_grid = np.meshgrid(*split_arrays, indexing='ij')
        return split_grid


    def mpi_print(self, *value):
        """Prints out using flush so it prints out immediately in an MPI
        setting."""
        print(*value, flush=True)


    def mpi_print_zero(self, *value):
        """Prints only at node rank = 0."""
        if self.rank == 0:
            self.mpi_print(*value)


    def send(self, data, to_rank=None, tag=11):
        """Sends data from current core to other specified or all cores.

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


    def recv(self, from_rank, tag=11):
        """Receive data from another node.

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


    def broadcast(self, data):
        """Broadcast data from rank=0 to all nodes."""
        if self.rank == 0:
            self.send(data, tag=11)
        else:
            data = self.recv(0, tag=11)
        self.wait()
        return data


    def send_up(self, data):
        datain = np.copy(data)
        """Send data from each node to the node above."""
        if self.rank < self.size-1:
            self.send(datain, to_rank=self.rank+1, tag=10+self.rank)
        if self.rank > 0:
            dataout = self.recv(self.rank-1, tag=10+self.rank-1)
        self.wait
        if self.rank == self.size-1:
            self.send(datain, to_rank=0, tag=10+self.size)
        if self.rank == 0:
            dataout = self.recv(self.size-1, tag=10+self.size)
        self.wait()
        return dataout


    def send_down(self, data):
        datain = np.copy(data)
        """Send data from each node to the node below."""
        if self.rank > 0:
            self.send(datain, to_rank=self.rank-1, tag=10+self.rank)
        if self.rank < self.size-1:
            dataout = self.recv(self.rank+1, tag=10+self.rank+1)
        self.wait()
        if self.rank == 0:
            self.send(datain, to_rank=self.size-1, tag=10+self.size)
        if self.rank == self.size-1:
            dataout = self.recv(0, tag=10+self.size)
        self.wait()
        return dataout


    def collect(self, data, outlist=False):
        """Collects a distributed data to the processor with rank=0.

        Parameters
        ----------
        data : array
            Distributed data set.
        """
        if np.isscalar(data):
            data = np.array([data])
        if self.rank == 0:
            datas = [data]
            for i in range(1, self.size):
                _data = self.recv(i, tag=10+i)
                datas.append(_data)
            if outlist is False:
                data = np.concatenate(datas)
            else:
                data = datas
        else:
            self.send(data, to_rank=0, tag=10+self.rank)
            data = None
        self.wait()
        return data


    def collect_noNone(self, data):
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


    def distribute(self, data):
        """Distribute and split data from rank 0.

        Parameters
        ----------
        data : array
            Full data set which will be split across the cores.
        """
        if self.rank == 0:
            split1, split2 = self.split(len(data))
            for i in range(1, len(split1)):
                self.send(data[split1[i]:split2[i]], to_rank=i, tag=10+i)
            data = data[split1[0]:split2[0]]
        else:
            data = self.recv(0, tag=10+self.rank)
        self.wait()
        return data


    def sum(self, data):
        """Sums a distributed data set to the processor with rank=0.

        Parameters
        ----------
        data : array
            distributed data set.
        """
        if self.rank == 0:
            for i in range(1, self.size):
                _data = self.recv(i, tag=10+i)
                data += _data
        else:
            self.send(data, to_rank=0, tag=10+self.rank)
            data = None
        self.wait()
        return data


    def mean(self, data):
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


    def min(self, data):
        mins = self.collect(np.min(data))
        if self.rank == 0:
            minval = np.min(mins)
            self.send(minval, tag=11)
        else:
            minval = self.recv(0, tag=11)
        self.wait()
        return minval


    def max(self, data):
        maxs = self.collect(np.max(data))
        if self.rank == 0:
            maxval = np.max(maxs)
            self.send(maxval, tag=11)
        else:
            maxval = self.recv(0, tag=11)
        self.wait()
        return maxval


    def end(self):
        """Ends MPI environment."""
        self.mpi.Finalize()
