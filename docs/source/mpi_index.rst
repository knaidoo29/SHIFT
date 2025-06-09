========
MPIutils
========

Installation and Basic Usage
----------------------------

``mpiutils`` is a submodule of ``SHIFT`` that enables ``MPI`` parallelisation in python.
To use this functionality you will need to install ``mpi4py`` (`link <https://mpi4py.readthedocs.io/en/stable/>`_).
Since correct installation requires careful linking of the Message Passing Interface to 
``mpi4py`` we leave the installation to the user and have choosen not to explicitly include 
``mpi4py`` as a dependency to ``SHIFT``. In this way the MPI functionality will remain an optional 
extra. 

To install ``mpi4py`` please follow the instructions `here <https://mpi4py.readthedocs.io/en/stable/install.html>`_.

Once ``mpi4py`` is installed the ``MPI`` utility class can be loading simply by writing:

.. code-block:: python

    from shift import mpiutils
    
    MPI = mpiutils.MPI()

To ensure python and ``numpy`` specifically do not carry out hidden parallelised processed in the background
we should first set the threads to ``1`` by including this at the top of every script:

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

The ``MPI`` object can now be used in any of the MPI enabled routines. For example to construct a 3D grid split across 
several processes we can use the ``shift.cart.mpi_grid3D`` function which we will write to a script called ``example_grid3D.py``.

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

    from shift import mpiutils
    
    MPI = mpiutils.MPI()

    boxsize = 100.
    ngrid = 512

    x3d, y3d, z3d = shift.cart.mpi_grid3D(boxsize, ngrid, MPI)

To run, simply do:

.. code-block:: bash

    mpirun -n 4 python example_grid3D.py

.. Note::

    ``mpirun`` should be replaced with the correct mpi exectuable, for instance on some
    HPCs this should be ``srun``.

Running with MPI
----------------

