# SHIFT : SpHerIcal Fourier Transforms

| | |      
|---------|----------------|
| Author  | Krishna Naidoo |   
| Version | 0.0.0          |
| Homepage | https://github.com/knaidoo29/SHIFT |
| Documentation | TBA |


## Introduction

SHIFT performs Fourier transforms of data in cartesian (using numpy wrappers), polar and spherical polar coordinates. SHIFT is mostly written in python but uses Fortran subroutines for additional speedup.

## Dependencies

* python 3
* numpy
* scipy
* f2py
* healpy

## Transforms

Currently implemented Fourier transforms.

1. FFT : Standard FFT wrappers for 1D, 2D and 3D.
2. PFT : Polar Fourier Transforms.

### To Do

1. Polar Fourier Transform
2. Spherical Fourier Transform
  * Spherical Bessel transform
  * Lado Samushia https://arxiv.org/pdf/1906.05866.pdf
3. Fast Fourier Transform wrapper in 1D, 2D and 3D.
  * sin
  * cos

### Functions

All functions in the module are listed below.

* `cart` : Cartesian FFT Transforms wrapping numpy functions.
  * `cart.forward_fft_1D` : Forward 1D FFT.
  * `cart.backward_fft_1D` : Backward 1D FFT.
  * `cart.forward_fft_2D` : Forward 2D FFT.
  * `cart.backward_fft_2D` : Backward 2D FFT.
  * `cart.forward_fft_3D` : Forward 3D FFT.
  * `cart.backward_fft_3D` : Backward 3D FFT.
  * `cart.grid1d` : Edges and bin centers for 1D grid.
  * `cart.grid2d` : Cartesian box grid in 2D.
  * `cart.grid3d` : Cartesian box grid in 3D.
  * `cart.get_fourier_grid_1D`: Get Fourier modes.
  * `cart.get_fourier_grid_2D`: Get 2D Fourier modes.
  * `cart.get_fourier_grid_3D`: Get 3D Fourier modes.
  * `cart.get_kf` : Fundamental frequency.
  * `cart.get_kn` : Nyquist frequency.

* `polar` : Polar Fourier Transforms.
  * `polar.get_n` : Returns the radial basis zeros order.
  * `polar.get_m` : Returns the angular basis order.
  * `polar.get_Jm` : Bessel function of order m.
  * `polar.get_dJm` : Derivative of the Bessel function of order m.
  * `polar.get_Jm_alt` : Reverse ordering of `get_Jm`.
  * `polar.get_dJm_alt` : Reverse ordering of `get_dJm`.
  * `polar.get_Jm_zeros` : Returns zeros of the Bessel function of order m.
  * `polar.get_dJm_zeros` : Returns zeros of the derivative of the Bessel function of order m.
  * `polar.get_Jm_large_zeros` : Returns large zeros of the Bessel function where scipy's zero-finder is unstable.
  * `polar.get_dJm_large_zeros` : Returns large zeros of the derivative Bessel function where scipy's zero-finder is unstable.
  * `polar.polargrid` : Creates a 2D polar grid.
  * `polar.wrap_polar` : Wraps polar grid in the angular axis for plotting purposes.
  * `polar.unwrap_polar` : Unwraps the polar grid.
  * `polar.get_knm` : Returns the Fourier modes for each Polar Fourier transform (PFT) mode.
  * `polar.get_Nnm_zero` : Returns the normalisation for the zero-boundary PFT.
  * `polar.get_Nnm_deri` : Returns the normalisation for the derivative-boundary PFT
  * `polar.get_Rnm` : Radial basis function for the PFT.
  * `polar.get_eix` : Euler's formula.
  * `polar.get_eix_star` : Complex conjugate of Euler's formula.
  * `polar.get_Phi_m` : Angular basis function for the PFT.
  * `polar.get_Phi_star_m` : Complex conjugate of the angular basis function for the PFT
  * `polar.get_Psi_nm` : PFT basis function.
  * `polar.get_Psi_star_nm` : Complex conjugate of PFT basis function.

* `src` : Fortran source code for improved performance.
  * `src.get_rnm` : Equivalent to `polar.get_Rnm` but implemented in Fortran.
  * `src.forward_half_pft` : Does half the forward PFT in Fortran.
  * `src.backward_half_pft` : Does half the backward PFT in Fortran.

* `utils` : Utility functions.
  * `utils.progress_bar` : For loop progression bar.
