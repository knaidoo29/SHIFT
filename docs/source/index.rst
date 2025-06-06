
.. image:: _static/SHIFT_logo_large_white.jpg
   :align: center
   :class: only-light

.. image:: _static/SHIFT_logo_large_black.jpg
   :align: center
   :class: only-dark
  
================================================
Scalable Helper Interface for Fourier Transforms
================================================

+---------------+-----------------------------------------+
| Author        | Krishna Naidoo                          |
+---------------+-----------------------------------------+
| Version       | 0.0.2                                   |
+---------------+-----------------------------------------+
| Repository    | https://github.com/knaidoo29/SHIFT      |
+---------------+-----------------------------------------+
| Documentation | https://shift.readthedocs.io            |
+---------------+-----------------------------------------+

.. warning ::
  SHIFT is currently in development. Functions and classes may change. Use with caution.

Introduction
============

SHIFT is a scalable interface library for computing FFTs in numpy. The library wraps numpy 
FFT routines and makes it easy to keep track of corresponding Fourier modes in Fourier 
space. The package can be scaled using MPI (using the mpi4py library), using a slab 
decomposition to perform distributed FFTs.

.. note ::

  SHIFT was originally designed to be a Spherical/Polar Fourier Transform library. This is 
  the origin of the original acronym for SHIFT (SpHerIcal Fourier Transforms). However, the 
  package has developed into being predominanty a helper and MPI interface for FFTs in 2D/3D 
  cartesian grids. The development of the Polar and Spherical Bessel transforms is still 
  ongoing but it no longer the focus.

Contents
========

.. toctree::
   :maxdepth: 2

   pft_index
   sht_index
   sbt_index
   cart_index
   api


Dependencies
============

* `numba <https://numba.pydata.org/>`_
* `numpy <http://www.numpy.org/>`_
* `scipy <https://scipy.org/>`_
* `healpy <https://healpy.readthedocs.io/>`_
* `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ [Optional: enables MPI parallelism]

Installation
============

SHIFT can be installed by cloning the github repository::

    git clone https://github.com/knaidoo29/SHIFT.git
    cd SHIFT
    python setup.py build
    python setup.py install

Once this is done you should be able to call SHIFT from python:

.. code-block:: python

    import shift

Support
=======

If you have any issues with the code or want to suggest ways to improve it please
open a new issue (`here <https://github.com/knaidoo29/SHIFT/issues>`_)
or (if you don't have a github account) email krishna.naidoo.11@ucl.ac.uk.


Version History
===============

**Version 0.0**:

  * Fast fourier transforms in cartesian coordinates up to 3D with additional utility functions.

  * Polar Fourier Transform with FFT used for the angular component while the radial component is calculated using Fortran source code.

  * Spherical Bessel Transform.
