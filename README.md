# SHIFT : SpHerIcal fast Fourier Transforms

[![Documentation Status](https://readthedocs.org/projects/shift-doc/badge/?version=latest)](https://shift-doc.readthedocs.io/en/latest/?badge=latest)

|              |                                    |
|--------------|------------------------------------|
|Author        | Krishna Naidoo                     |
|Version       | 0.0.1                              |
|Documentation | https://shift-doc.readthedocs.io/  |
|Repository    | https://github.com/knaidoo29/SHIFT |

> **WARNING**: SHIFT is currently in development. Functions and classes may change. Use with caution.

## Introduction

SHIFT performs Fast Fourier transforms on data in polar, spherical and cartesian coordinates. SHIFT is written mostly in python but uses Fortran subroutines for heavy computation and speed.

MPI functionality can be enabled through the installation of the python library mpi4py but will require the additional installation of MPIutils which handles all of the MPI enabled functions. The class MPI is passed as an additional argument for parallelisation.

## Dependencies

SHIFT is being developed in Python 3.8 but should work on all versions >3.4. SHIFT is written mostly in python but the heavy computation is carried out in Fortran. Compiling the Fortran source code will require the availability of a fortran compiler usually gfortran (which comes with gcc).

The following Python modules are required.

* `numpy`
* `scipy`
* `healpy`

If you want to run with MPI you will need the following:

* `mpi4py`
* `MPIutils`

For testing you will require `nose` or `pytest`.

## Installation

SHIFT can be installed by cloning the github repository:

```
git clone https://github.com/knaidoo29/SHIFT.git
cd SHIFT
python setup.py build
python setup.py install
```

Once this is done you should be able to call SHIFT from python:


```python
import shift
```

## Documentation

In depth documentation and tutorials are provided [here](https://shift-doc.readthedocs.io/).

## Tutorials

TBA

## Citing

TBA

## Support

If you have any issues with the code or want to suggest ways to improve it please open a new issue ([here](https://github.com/knaidoo29/SHIFT/issues))
or (if you don't have a github account) email _krishna.naidoo.11@ucl.ac.uk_.
