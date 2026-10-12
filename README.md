![biglogo](https://raw.githubusercontent.com/knaidoo29/SHIFT/main/docs/source/_static/SHIFT_logo_large_github.jpg)

<p align="center">
    <a href="https://github.com/knaidoo29/SHIFT/actions/workflows/tests.yml">
    <img src="https://github.com/knaidoo29/SHIFT/actions/workflows/tests.yml/badge.svg" alt="Python Tests">
    </a>
    <a href="https://codecov.io/gh/knaidoo29/SHIFT" > 
    <img src="https://codecov.io/gh/knaidoo29/SHIFT/graph/badge.svg?token=HX4OB31MS3"/> 
    </a>
    <a href="https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue">
    <img src="https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue" alt="Python Version Support">
    </a>
    <a href="https://img.shields.io/github/v/release/knaidoo29/SHIFT">  
    <img src="https://img.shields.io/github/v/release/knaidoo29/SHIFT" alt="Version">
    </a>
    <a href="https://pypi.org/project/shift-fft/">
    <img src="https://img.shields.io/pypi/v/shift-fft.svg" alt="PyPI version">
    </a>
    <a href="https://shift.readthedocs.io/en/latest/">
    <img src="https://readthedocs.org/projects/shift/badge/?version=latest" alt="Documentation Status">
    </a>
    <a href="https://github.com/knaidoo29/SHIFT">
    <img src="https://img.shields.io/badge/GitHub-repo-blue?logo=github" alt="GitHub repository">
    </a>
    <a href="https://img.shields.io/github/stars/knaidoo29/shift">
    <img src="https://img.shields.io/github/stars/knaidoo29/shift" alt="github: stars">
    </a>
    <a href="https://img.shields.io/github/stars/knaidoo29/shift">
    <img src="https://img.shields.io/github/forks/knaidoo29/shift" alt="github: forks">
    </a>
    <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    </a>
    <a href="https://doi.org/10.5281/zenodo.17093446">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17093446.svg" alt="zenodo: DOI">
    </a>
</p>

# SHIFT : Scalable High-level Interface for Fourier Transforms

## Introduction

SHIFT is a lightweight Python library designed to simplify Fourier
transforms and spectral calculations on regular grids. Built around
SciPy's FFT routines, SHIFT provides a convenient interface for
computing Fourier, discrete cosine (DCT), and discrete sine (DST)
transforms in one, two, and three dimensions, alongside utilities
for constructing grids and their corresponding Fourier-space modes.

A central motivation behind SHIFT is to make parallel Fourier
transforms easier to install, use, and integrate into scientific
applications. Unlike implementations requiring FFTW and additional
compiled extensions, SHIFT uses SciPy's FFT backend and supports
distributed transforms through ``mpi4py``, employing an MPI slab
decomposition. This avoids the need for a separate FFTW installation
while retaining compatibility with standard scientific Python
environments.

In addition to spectral transforms, SHIFT provides utilities for
MPI communication and distributed array operations, making it
suitable for integration into larger scientific computing workflows.

> **NOTE**: SHIFT was originally conceived as a framework for Fourier
  transforms in Cartesian, polar, and spherical coordinate systems.
  Current development focuses on Cartesian spectral transforms and
  MPI-based scientific computing. Experimental polar and spherical
  transform implementations are available in the ``sphere`` branch,
  but are not part of the supported release and should be used
  with caution. Further development of these capabilities is
  planned.

## Dependencies

* `numba`
* `numpy`
* `scipy`
* `mpi4py` [Optional: enables MPI distributed processes]

## Installation

SHIFT can be installed via `pip`:

```
pip install shift-fft
```

Or by cloning the github repository:

```
git clone https://github.com/knaidoo29/SHIFT.git
cd SHIFT
pip install .
```

The latter of which can be used to install specific branches. 

Once this is done you should be able to call `SHIFT` from python:

```python
import shift
```

To use the `MPI` functionality please take a look at the documentation which instructs users how to use the `SHIFT` `MPI` object and how to run these distributed jobs successfully without errors or MPI related hanging.

## Documentation

In depth documentation and tutorials are provided [here](https://shift.readthedocs.io/).

## License

SHIFT is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use **SHIFT** in your work, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17093446.svg)](https://doi.org/10.5281/zenodo.17093446)

```bibtex
  @software{naidoo_shift_2025,
    author       = {Naidoo, Krishna},
    title        = {SHIFT: a scalable MPI library for computing fast Fourier transforms in python},
    year         = 2025,
    publisher    = {Zenodo},
    doi          = {10.5281/zenodo.17093446},
    url          = {https://doi.org/10.5281/zenodo.17093446}}
```

## Support

If you have any issues with the code or want to suggest ways to improve it please open a new issue ([here](https://github.com/knaidoo29/SHIFT/issues))
or (if you don't have a github account) email _krishna.naidoo.11@ucl.ac.uk_.
