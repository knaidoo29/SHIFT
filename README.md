![biglogo](docs/source/_static/SHIFT_logo_large_github.jpg)

# SHIFT : Scalable Helper Interface for Fourier Transforms

[![Documentation Status](https://readthedocs.org/projects/shift/badge/?version=latest)](https://shift.readthedocs.io/en/latest/?badge=latest)

## Introduction

SHIFT is a scalable interface library for computing FFTs in scipy. The library wraps scipy FFT routines and makes it easy to keep track of corresponding Fourier modes in Fourier space. The package can be scaled using MPI (using the mpi4py library), using a slab decomposition to perform distributed FFTs.

> **NOTE**: SHIFT was originally designed to be a Spherical/Polar Fourier Transform library. This is the origin of the original acronym for SHIFT (SpHerIcal Fourier Transforms). However, the package has developed into being predominanty a helper and MPI interface for FFTs in 2D/3D cartesian grids. The development of the Polar and Spherical Bessel transforms is still ongoing but it no longer the focus.

## Dependencies

* `numba`
* `numba-scipy`
* `numpy`
* `scipy`
* `healpy`
* `mpi4py` [Optional: enables MPI parallelism]

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

In depth documentation and tutorials are provided [here](https://shift.readthedocs.io/).

## Tutorials

TBA

## Citing

TBA

## Support

If you have any issues with the code or want to suggest ways to improve it please open a new issue ([here](https://github.com/knaidoo29/SHIFT/issues))
or (if you don't have a github account) email _krishna.naidoo.11@ucl.ac.uk_.
