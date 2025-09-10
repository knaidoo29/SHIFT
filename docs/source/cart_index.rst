================================
Cartesian Fast Fourier Transform
================================

Defining a cartesian grid
=========================

Before we compute the Fourier transforms it is useful to define the grid the Fourier 
transform will be computed on. The utility of this will become much more apparent 
when we go into Fourier space.

First let's define the size of our box (``boxsize``) and the resolution of the grid
(``ngrid``).

.. Note:: 

  Both the ``boxsize`` and ``ngrid`` can either be single numbers meaning the box is 
  actually a square/cube or a list for each axes if you would like a rectagle/cuboid shape.

.. code-block:: python

  import shift

  boxsize = 100.
  ngrid = 128

  xedged, x = shift.cart.grid1D(boxsize, ngrid)

Where xedges tells us the ``xedges`` of the bin and ``x`` the centers.

In 2D and 3D this can be defined as

.. code-block:: python

  # in 2D
  x2D, y2D = shift.cart.grid2D(boxsize, ngrid)

  # in 3D
  x3D, y3D, z3D = shift.cart.grid3D(boxsize, ngrid)

Fourier transforms forward/backwards
====================================

Let's now define some field which we call ``fgrid`` which is defined in the same
cartesian grid we defined above. Fourier transforming via an FFT is simple:

.. code-block:: python

  # in 1D
  fkgrid = shift.cart.fft1D(fgrid, boxsize)

  # in 2D 
  fkgrid = shift.cart.fft2D(fgrid, boxsize)

  # in 3D 
  fkgrid = shift.cart.fft3D(fgrid, boxsize)

We can find the corresponding Fourier modes to this array using ``kgrid`` functions like so

.. code-block:: python

  # in 1D
  kx1D = shift.cart.kgrid1D(boxsize, ngrid)

  # in 2D
  kx2D, ky2D = shift.cart.kgrid2D(boxsize, ngrid)

  # in 3D
  kx3D, ky3D, kz3D = shift.cart.kgrid3D(boxsize, ngrid)

Computations involving the Fourier modes ``k`` can now be easily computed. Once the analysis
in Fourier space is complete, we can return to real space by running

.. code-block:: python

  # in 1D
  fgrid = shift.cart.ifft1D(fkgrid, boxsize)

  # in 2D 
  fgrid = shift.cart.ifft2D(fkgrid, boxsize)

  # in 3D 
  fgrid = shift.cart.ifft3D(fkgrid, boxsize)

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

  # in 1D
  fkgrid = shift.cart.dct1D(fgrid, boxsize, type=2)

  # in 2D 
  fkgrid = shift.cart.dct2D(fgrid, boxsize, type=2)

  # in 3D 
  fkgrid = shift.cart.dct3D(fgrid, boxsize, type=2)

  # BACKWARD DCT Transform

  # in 1D
  fgrid = shift.cart.idct1D(fkgrid, boxsize, type=2)

  # in 2D 
  fgrid = shift.cart.idct2D(fkgrid, boxsize, type=2)

  # in 3D 
  fgrid = shift.cart.idct3D(fkgrid, boxsize, type=2)

  # Discrete Sine Transform

  # FORWARD DST Transform

  # in 1D
  fkgrid = shift.cart.dst1D(fgrid, boxsize, type=2)

  # in 2D 
  fkgrid = shift.cart.dst2D(fgrid, boxsize, type=2)

  # in 3D 
  fkgrid = shift.cart.dst3D(fgrid, boxsize, type=2)

  # BACKWARD DST Transform

  # in 1D
  fgrid = shift.cart.idst1D(fkgrid, boxsize, type=2)

  # in 2D 
  fgrid = shift.cart.idst2D(fkgrid, boxsize, type=2)

  # in 3D 
  fgrid = shift.cart.idst3D(fkgrid, boxsize, type=2)

.. Note:: 

  The default is ``type=2``. This can be changed and follows the scipy definitions for DCT/DST. Just 
  ensure you use the same ``type`` for both the forwards and backward transforms.

The corresponding Fourier grid is slightly different for the DCT/DST and can be accessed by running

.. code-block:: python

  # Discrete Cosine Transform

  # in 1D
  kx1D = shift.cart.kgrid1D_dct(boxsize, ngrid)

  # in 2D
  kx2D, ky2D = shift.cart.kgrid2D_dct(boxsize, ngrid)

  # in 3D
  kx3D, ky3D, kz3D = shift.cart.kgrid3D_dct(boxsize, ngrid)

  # Discrete Sine Transform

  # in 1D
  kx1D = shift.cart.kgrid1D_dst(boxsize, ngrid)

  # in 2D
  kx2D, ky2D = shift.cart.kgrid2D_dst(boxsize, ngrid)

  # in 3D
  kx3D, ky3D, kz3D = shift.cart.kgrid3D_dst(boxsize, ngrid)

Cartesian FFT API
=================

.. toctree::
  :maxdepth: 2

  api_cart

