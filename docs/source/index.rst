============================
SpHerIcal Fourier Transforms
============================


+---------------+-----------------------------------------+
| Author        | Krishna Naidoo                          |
+---------------+-----------------------------------------+
| Version       | 0.0.1                                   |
+---------------+-----------------------------------------+
| Repository    | https://github.com/knaidoo29/SHIFT     |
+---------------+-----------------------------------------+
| Documentation | https://shift-docs.readthedocs.io/     |
+---------------+-----------------------------------------+


Introduction
============

SHIFT performs Fourier transforms of data in cartesian (using functions that wrap
scipy FFT functions), polar and spherical polar coordinates. SHIFT is mostly written
in python but uses Fortran subroutines for heavy computation and speed.

MPI functionality can be enabled through the installation of the python library
mpi4py but will require the additional installation of MPIutils which handles
all of the MPI enabled functions. The class MPI is passed as an additional argument
for parallelisation.


Contents
--------

.. toctree::
   :maxdepth: 1

   pft_index
   sht_index
   sbt_index
   cart_index
   api


Dependencies
------------

SHIFT is being developed in Python 3.8 but should work on all versions >3.4. SHIFT
is written mostly in python but the heavy computation is carried out in Fortran.
Compiling the Fortran source code will require the availability of a fortran compiler
usually gfortran (which comes with gcc).

The following Python modules are required.

* `numpy <http://www.numpy.org/>`_
* `scipy <https://scipy.org/>`_

For testing you will require `nose <https://nose.readthedocs.io/en/latest/>`_ or
`pytest <http://pytest.org/en/latest/>`_ .


Installation
------------

SHIFT can be installed by cloning the github repository::

    git clone https://github.com/knaidoo29/SHIFT.git
    cd SHIFT
    python setup.py build
    python setup.py install

Once this is done you should be able to call SHIFT from python:

.. code-block:: python

    import shift


Support
-------

If you have any issues with the code or want to suggest ways to improve it please
open a new issue (`here <https://github.com/knaidoo29/SHIFT/issues>`_)
or (if you don't have a github account) email krishna.naidoo.11@ucl.ac.uk.


Version History
---------------

**Version 0.0**:

  * Fast fourier transforms in cartesian coordinates up to 3D with additional utility functions.

  * Polar Fourier Transform with FFT used for the angular component while the radial component is calculated using Fortran source code.

  * Spherical Bessel Transform.
