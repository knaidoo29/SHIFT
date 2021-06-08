# SHIFT : SpHerIcal Fourier Transforms

| | |      
|---------|----------------|
| Author  | Krishna Naidoo |   
| Version | 0.0.0          |
| Homepage | https://github.com/knaidoo29/SHIFT |
| Documentation | TBA |


## Introduction

SHIFT performs Fourier transforms of data in 2D, 3D, polar and spherical polar coordinates. SHIFT is mostly written in python but also includes MPI routines
for computations on large data sets.

## Dependencies

* python 3
* numpy
* scipy
* f2py
* healpy

## Transforms

### To Do

1. Fast Fourier Transform wrapper in 1D, 2D and 3D.
  * normal
  * sin
  * cos
2. Polar Fourier Transform
3. Spherical Fourier Transform
  * Spherical Bessel transform
  * Lado Samushia https://arxiv.org/pdf/1906.05866.pdf

### Functions

* `cart` : Cartesian FFT Transforms wrapping numpy functions.
  * `cart.forward_fft_1D` : Forward 1D FFT.
  * `cart.backward_fft_1D` : Backward 1D FFT.
  * `cart.forward_fft_2D` : Forward 2D FFT.
  * `cart.backward_fft_2D` : Backward 2D FFT.
  * `cart.forward_fft_3D` : Forward 3D FFT.
  * `cart.backward_fft_3D` : Backward 3D FFT.
  * `cart.get_fourier_grid_1D`: Get Fourier modes.
  * `cart.get_fourier_grid_2D`: Get 2D Fourier modes.
  * `cart.get_fourier_grid_3D`: Get 3D Fourier modes.
  * `cart.get_kf` : Fundamental frequency.
  * `cart.get_kn` : Nyquist frequency.
