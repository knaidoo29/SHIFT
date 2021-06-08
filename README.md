# SHIFT : SpHerIcal Fourier Transforms

Author :  Krishna Naidoo
Version : 0.0.0
Homepage : https://github.com/knaidoo29/PolarFT
Documentation : TBA

## Introduction

PolarFT performs Fourier transforms of data in 2D polar and 3D spherical polar coordinates. PolarFT is mostly written in python but also includes MPI routines
for computations on large data sets.

## Dependencies

* python 3
* numpy
* scipy
* f2py
* healpy

## Transforms

1. Fast Fourier Transform wrapper in 1D, 2D and 3D.
  * normal
  * sin
  * cos
2. Polar Fourier Transform
3. Spherical Fourier Transform
  * Spherical Bessel transform
  * Lado Samushia https://arxiv.org/pdf/1906.05866.pdf
