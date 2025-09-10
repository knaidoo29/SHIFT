====================================
MPI Cartesian Fast Fourier Transform
====================================

The `MPI` Object and how to run `MPI` processes
=================================================

MPI enables us to run processes across multiply processors. When writing an MPI python script, remember
that each process will see the same script and run through that same script. To get the benefits of MPI
you must split the data you are working on or ensure that each process (called a `rank` in MPI) works on
different pieces of the data. With this in mind `SHIFT` will construct 2D/3D grids across multiple processors.
The FFT routines will be computed using a slab decomposition, meaning the grid is divided along the x-axis
in real space and along the y-axis in Fourier space. To run `SHIFT` with `MPI` we need to make use of the 
`MPI` object located in `shift.mpiutils`. This is discussed in :doc:`MPIutils <mpiutils>` guide and should 
be read before going any further.

The TL;DR, each MPI script should being with the following:

.. code-block:: python

  import sys
  from os import environ

  # Set thread environment variables FIRST
  N_THREADS = '1'
  environ['OMP_NUM_THREADS'] = N_THREADS
  environ['OPENBLAS_NUM_THREADS'] = N_THREADS
  environ['MKL_NUM_THREADS'] = N_THREADS
  environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
  environ['NUMEXPR_NUM_THREADS'] = N_THREADS

  import shift
  
  MPI = shift.mpiutils.MPI()

and should end with:

.. code-block:: python

  MPI.end()

Which closes the MPI environment before ending the script.

To ensure MPI processes are synced use the wait functionality

.. code-block:: python

  MPI.wait()

To run the script with 4 processors:

.. code-block:: bash

  mpirun -n 4 python mpi_script.py

.. Note::

  ``mpirun`` should be replaced with the correct mpi exectuable, for instance on some
  HPCs this should be ``srun``.


Defining a distributed cartesian grid
=====================================

Before we compute the Fourier transforms it is useful to define the grid the Fourier 
transform will be computed on. The utility of this will become much more apparent 
when we go into Fourier space.

First let's define the size of our box (``boxsize``) and the resolution of the grid
(``ngrid``).

.. Note:: 

  Both the ``boxsize`` and ``ngrid`` can either be single numbers meaning the box is 
  actually a square/cube or a list for each axes if you would like a rectagle/cuboid shape.

.. code-block:: python

  boxsize = 100.
  ngrid = 128

  xedged, x = shift.cart.mpi_grid1D(boxsize, ngrid, MPI)

Where xedges tells us the ``xedges`` of the bin and ``x`` the centers.

.. Note::

  Notice the MPI object is entered as an argument here. This enables the code to distribute
  the array to the specific MPI setup.

In 2D and 3D this can be defined as

.. code-block:: python

  # in 2D
  x2D, y2D = shift.cart.mpi_grid2D(boxsize, ngrid, MPI)

  # in 3D
  x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, MPI)

Each process or `MPI.rank` sees a different part of the grid.

MPI Fourier transforms forward/backwards
========================================

Let's now define some field which we call ``fgrid`` which is defined in the same
cartesian grid we defined above. You must ensure that ``fgrid`` has the same shape 
as the real grids defined above -- easy to setup if they are directly related. Fourier 
transforming this data via an FFT is simple:

.. code-block:: python

  # in 2D 
  fkgrid = shift.cart.mpi_fft2D(fgrid, boxsize, ngrid, MPI)

  # in 3D 
  fkgrid = shift.cart.mpi_fft3D(fgrid, boxsize, ngrid, MPI)

.. Note::

  Because we are using slab decomposition we do not support FFTs of 1D grids.

.. Note::
  
  The FFT functions now require the global ``ngrid`` as an input as well as the ``MPI`` object.

We can find the corresponding Fourier modes to this array using the ``kgrid`` functions like so

.. code-block:: python

  # in 1D
  kx1D = shift.cart.mpi_kgrid1D(boxsize, ngrid)

  # in 2D
  kx2D, ky2D = shift.cart.mpi_kgrid2D(boxsize, ngrid)

  # in 3D
  kx3D, ky3D, kz3D = shift.cart.mpi_kgrid3D(boxsize, ngrid)

Computations involving the Fourier modes ``k`` can now be easily computed. Once the analysis
in Fourier space is complete, we can return to real space by running

.. code-block:: python

  # in 2D 
  fgrid = shift.cart.mpi_ifft2D(fkgrid, boxsize, ngrid, MPI)

  # in 3D 
  fgrid = shift.cart.mpi_ifft3D(fkgrid, boxsize, ngrid, MPI)

For FFTs using the Discrete Cosine or Sine Transform the analysis can be repeated in a similar
fashion.

.. Note:: 

  The DCT/DST produce almost identical results to the FFT, except near the boundaries where the
  dirichlet and neumann boundary conditions will enforce zeros for the DST and zero derivatives 
  in DCT. Keep this in mind and depending on your setup, you may choose to ignore these boundaries 
  from the analysis.

.. code-block:: python
  
  # Discrete Cosine Transform

  # FORWARD DCT Transform

  # in 2D 
  fkgrid = shift.cart.mpi_dct2D(fgrid, boxsize, ngrid, MPI, type=2)

  # in 3D 
  fkgrid = shift.cart.mpi_dct3D(fgrid, boxsize, ngrid, MPI, type=2)

  # BACKWARD DCT Transform

  # in 2D 
  fgrid = shift.cart.mpi_idct2D(fkgrid, boxsize, ngrid, MPI, type=2)

  # in 3D 
  fgrid = shift.cart.mpi_idct3D(fkgrid, boxsize, ngrid, MPI, type=2)

  # Discrete Sine Transform

  # FORWARD DST Transform

  # in 2D 
  fkgrid = shift.cart.mpi_dst2D(fgrid, boxsize, ngrid, MPI, type=2)

  # in 3D 
  fkgrid = shift.cart.mpi_dst3D(fgrid, boxsize, ngrid, MPI, type=2)

  # BACKWARD DST Transform

  # in 2D 
  fgrid = shift.cart.mpi_idst2D(fkgrid, boxsize, ngrid, MPI, type=2)

  # in 3D 
  fgrid = shift.cart.mpi_idst3D(fkgrid, boxsize, ngrid, MPI, type=2)

.. Note:: 

  The default is ``type=2``. This can be changed and follows the scipy definitions for DCT/DST. Just 
  ensure you use the same ``type`` for both the forwards and backward transforms.

The corresponding Fourier grid is slightly different for the DCT/DST and can be accessed by running

.. code-block:: python

  # Discrete Cosine Transform

  # in 2D
  kx2D, ky2D = shift.cart.mpi_kgrid2D_dct(boxsize, ngrid, MPI)

  # in 3D
  kx3D, ky3D, kz3D = shift.cart.mpi_kgrid3D_dct(boxsize, ngrid, MPI)

  # Discrete Sine Transform

  # in 2D
  kx2D, ky2D = shift.cart.mpi_kgrid2D_dst(boxsize, ngrid, MPI)

  # in 3D
  kx3D, ky3D, kz3D = shift.cart.mpi_kgrid3D_dst(boxsize, ngrid, MPI)

Cartesian MPI FFT API
=====================

.. toctree::
  :maxdepth: 2

  api_mpi_cart

