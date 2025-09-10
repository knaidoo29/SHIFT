cart
====

The following functions are MPI enabled.

Convolutions
------------

.. autofunction:: shift.cart.convolve_gaussian

Differentiation
---------------

.. autofunction:: shift.cart.dfdk
.. autofunction:: shift.cart.dfdk2

Cartesian grids with MPI
------------------------

.. autofunction:: shift.cart.mpi_grid1D
.. autofunction:: shift.cart.mpi_grid2D
.. autofunction:: shift.cart.mpi_grid3D

FFT with MPI
------------

Fourier grids
^^^^^^^^^^^^^

.. autofunction:: shift.cart.mpi_kgrid1D
.. autofunction:: shift.cart.mpi_kgrid2D
.. autofunction:: shift.cart.mpi_kgrid3D

Forward/Backward Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: shift.cart.mpi_fft2D
.. autofunction:: shift.cart.mpi_ifft2D
.. autofunction:: shift.cart.mpi_fft3D
.. autofunction:: shift.cart.mpi_ifft3D

DCT with MPI
------------

Fourier grids
^^^^^^^^^^^^^

.. autofunction:: shift.cart.mpi_kgrid1D_dct
.. autofunction:: shift.cart.mpi_kgrid2D_dct
.. autofunction:: shift.cart.mpi_kgrid3D_dct

Forward/Backward Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: shift.cart.mpi_dct2D
.. autofunction:: shift.cart.mpi_idct2D
.. autofunction:: shift.cart.mpi_dct3D
.. autofunction:: shift.cart.mpi_idct3D

DST with MPI
------------

Fourier grids
^^^^^^^^^^^^^

.. autofunction:: shift.cart.mpi_kgrid1D_dst
.. autofunction:: shift.cart.mpi_kgrid2D_dst
.. autofunction:: shift.cart.mpi_kgrid3D_dst

Forward/Backward Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: shift.cart.mpi_dst2D
.. autofunction:: shift.cart.mpi_idst2D
.. autofunction:: shift.cart.mpi_dst3D
.. autofunction:: shift.cart.mpi_idst3D

.. Power Spectrum
.. --------------

.. .. autofunction:: shift.cart.mpi_get_pofk_2D
.. .. autofunction:: shift.cart.mpi_get_pofk_3D