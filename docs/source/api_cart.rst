cart
====

Convolutions
------------

.. autofunction:: shift.cart.convolve_gaussian

Differentiation
---------------

.. autofunction:: shift.cart.dfdk
.. autofunction:: shift.cart.dfdk2

Cartesian grids
---------------

.. autofunction:: shift.cart.grid1D
.. autofunction:: shift.cart.grid2D
.. autofunction:: shift.cart.grid3D

Cartesian grids with MPI
------------------------

.. autofunction:: shift.cart.mpi_grid1D
.. autofunction:: shift.cart.mpi_grid2D
.. autofunction:: shift.cart.mpi_grid3D

FFT
---

Fourier grids
^^^^^^^^^^^^^

.. autofunction:: shift.cart.kgrid1D
.. autofunction:: shift.cart.kgrid2D
.. autofunction:: shift.cart.kgrid3D

Forward/Backward Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: shift.cart.fft1D
.. autofunction:: shift.cart.ifft1D
.. autofunction:: shift.cart.fft2D
.. autofunction:: shift.cart.ifft2D
.. autofunction:: shift.cart.fft3D
.. autofunction:: shift.cart.ifft3D

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

DCT
---

Fourier grids
^^^^^^^^^^^^^

.. autofunction:: shift.cart.kgrid1D_dct
.. autofunction:: shift.cart.kgrid2D_dct
.. autofunction:: shift.cart.kgrid3D_dct

Forward/Backward Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: shift.cart.dct1D
.. autofunction:: shift.cart.idct1D
.. autofunction:: shift.cart.dct2D
.. autofunction:: shift.cart.idct2D
.. autofunction:: shift.cart.dct3D
.. autofunction:: shift.cart.idct3D

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

DST
---

Fourier grids
^^^^^^^^^^^^^

.. autofunction:: shift.cart.kgrid1D_dst
.. autofunction:: shift.cart.kgrid2D_dst
.. autofunction:: shift.cart.kgrid3D_dst

Forward/Backward Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: shift.cart.dst1D
.. autofunction:: shift.cart.idst1D
.. autofunction:: shift.cart.dst2D
.. autofunction:: shift.cart.idst2D
.. autofunction:: shift.cart.dst3D
.. autofunction:: shift.cart.idst3D

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

Multiply
--------

.. autofunction:: shift.cart.mult_fk_2D
.. autofunction:: shift.cart.mult_fk_3D

Power Spectrum
--------------

.. autofunction:: shift.cart.get_pofk_2D
.. autofunction:: shift.cart.get_pofk_3D

Utility Functions
-----------------

.. autofunction:: shift.cart.get_kf
.. autofunction:: shift.cart.get_kn
.. autofunction:: shift.cart.fftshift
.. autofunction:: shift.cart.ifftshift
.. autofunction:: shift.cart.normalise_freq
.. autofunction:: shift.cart.unnormalise_freq