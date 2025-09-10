import numpy as np

from . import bessel
from . import grid
from . import pft_basis

from .. import cart
from .. import src
from .. import utils


class PFT:

    def __init__(self) -> None:
        """
        Initialises PFT class.
        """
        self.Nr = None
        self.Np = None
        self.Rmax = None
        self.p2d = None
        self.r2d = None
        self.dr = None
        self.dphi = None
        self.m = None
        self.n = None
        self.m2d = None
        self.n2d = None
        self.xnm = None
        self.knm = None
        self.Nnm = None
        self.m2d_flat = None
        self.n2d_flat = None
        self.xnm_flat = None
        self.knm_flat = None
        self.Nnm_flat = None

    def prep(
        self,
        Nr: int,
        Np: int,
        Rmax: float,
        boundary: str = "zero",
        nswitch: int = 2,
        mswitch: int = 20,
    ) -> None:
        """
        Prepare PFT for Polar Fourier Transform

        Parameters
        ----------
        Nr : int
            Length of radial axis.
        Np : int
            Length of angular axis.
        Rmax : float
            Radius length.
        boundary : str, optional
            Boundary condition, zero or derivative boundary condition
        nswitch : int, optional
            Uses scipy to find zeros before nswitch and then uses minimization functions to find them.
        mswitch : int, optional
            Uses zeros at m-2 and m-1 to predict approximations for m and then minimizes.
        """
        self.Nr = Nr
        self.Np = Np
        self.Rmax = Rmax
        self.boundary = boundary
        self.p2d, self.r2d = grid.polargrid(self.Rmax, self.Nr, self.Np)
        self.dr = self.r2d[0][1] - self.r2d[0][0]
        self.dphi = self.p2d[1][0] - self.p2d[0][0]
        self.m = pft_basis.get_m(self.Np)
        self.n = pft_basis.get_n(self.Nr)
        self.m2d, self.n2d = np.meshgrid(self.m, self.n, indexing="ij")
        self.xnm = np.zeros(np.shape(self.m2d))
        self.knm = np.zeros(np.shape(self.m2d))
        self.Nnm = np.zeros(np.shape(self.m2d))
        lenm = len(self.m2d)
        for i in range(0, int(lenm / 2) + 1):
            mval = self.m[i]
            nval = self.n[-1]
            if self.boundary == "zero":
                if abs(mval) < mswitch:
                    _xnm = bessel.get_Jm_large_zeros(mval, nval)
                else:
                    _xnm1 = self.xnm[abs(mval) - 1]
                    _xnm2 = self.xnm[abs(mval) - 2]
                    _xnm = bessel.get_Jm_large_zeros(
                        mval, nval, nstop=nswitch, xnm2=_xnm2, xnm1=_xnm1
                    )
                _knm = pft_basis.get_knm(_xnm, self.Rmax)
                _Nnm = pft_basis.get_Nnm_zero(mval, _xnm, self.Rmax)
            else:
                if abs(mval) < mswitch:
                    _xnm = bessel.get_dJm_large_zeros(mval, nval)
                else:
                    _xnm1 = self.xnm[abs(mval) - 1]
                    _xnm2 = self.xnm[abs(mval) - 2]
                    _xnm = bessel.get_dJm_large_zeros(
                        mval, nval, nstop=nswitch, xnm2=_xnm2, xnm1=_xnm1
                    )
                _knm = pft_basis.get_knm(_xnm, self.Rmax)
                _Nnm = pft_basis.get_Nnm_deri(mval, _xnm, self.Rmax)
            self.xnm[i] = _xnm
            self.knm[i] = _knm
            self.Nnm[i] = _Nnm
            if i != lenm - i and i != 0:
                self.xnm[lenm - i] = _xnm
                self.knm[lenm - i] = _knm
                self.Nnm[lenm - i] = _Nnm
            utils.progress_bar(i, int(lenm / 2) + 1)
        self.m2d_flat = np.copy(self.m2d).flatten()
        self.n2d_flat = np.copy(self.n2d).flatten()
        self.xnm_flat = np.copy(self.xnm).flatten()
        self.knm_flat = np.copy(self.knm).flatten()
        self.Nnm_flat = np.copy(self.Nnm).flatten()

    def _benchmark_forward(self, f: np.ndarray) -> np.ndarray:
        """
        Computes the polar Fourier coefficients using the benchmark function,
        meaning this computes the coefficients with the double integral and no
        extra optimisations.

        Parameters
        ----------
        f : 2darray
            Field values on a 2D polar grid.

        Returns
        -------
        Pnm : 2darray
            Polar fourier coefficients.
        """
        Pnm = np.zeros(len(self.m2d_flat)) + 1j * np.zeros(len(self.m2d_flat))
        for i in range(0, len(self.m2d_flat)):
            Psi = pft_basis.get_Psi_star_nm(
                self.n2d_flat[i],
                self.m2d_flat[i],
                self.r2d,
                self.p2d,
                self.knm_flat[i],
                self.Nnm_flat[i],
            )
            Pnm.real[i] = np.sum(f * Psi.real * self.r2d) * self.dr * self.dphi
            Pnm.imag[i] = np.sum(f * Psi.imag * self.r2d) * self.dr * self.dphi
            utils.progress_bar(i, len(self.m2d_flat))
        Pnm = Pnm.reshape(np.shape(self.m2d))
        return Pnm

    def _half_fft_forward(self, f: np.ndarray) -> np.ndarray:
        """
        Computes the polar Fourier coefficients using FFT for the angular components
        which means each coefficient is calculated from a single integral rather than a
        double integral.

        Parameters
        ----------
        f : 2darray
            Field values on a 2D polar grid.

        Returns
        -------
        Pnm : 2darray
            Polar fourier coefficients.
        """
        Pm = cart.fft1D(f, 2 * np.pi, axis=0)
        Pnm = np.zeros(len(self.m2d_flat)) + 1j * np.zeros(len(self.m2d_flat))
        r = self.r2d[0]
        for i in range(0, len(self.m2d_flat)):
            Rnm = pft_basis.get_Rnm(
                r, self.m2d_flat[i], self.knm_flat[i], self.Nnm_flat[i]
            )
            if self.m2d_flat[i] < 0:
                m_index = self.Np + self.m2d_flat[i]
            else:
                m_index = self.m2d_flat[i]
            Pnm.real[i] = np.sum(r * Rnm * Pm.real[m_index]) * self.dr
            Pnm.imag[i] = np.sum(r * Rnm * Pm.imag[m_index]) * self.dr
            utils.progress_bar(i, len(self.m2d_flat))
        Pnm = Pnm.reshape(np.shape(self.m2d))
        return Pnm

    def _half_fft_half_numba_forward(self, f: np.ndarray) -> np.ndarray:
        """
        Computes the polar Fourier coefficients using FFT for the angular components
        and numba for the rest.

        Parameters
        ----------
        f : 2darray
            Field values on a 2D polar grid.

        Returns
        -------
        Pnm : 2darray
            Polar fourier coefficients.
        """
        Pm = cart.fft1D(f, 2 * np.pi, axis=0)
        Pm = Pm.flatten()
        r = self.r2d[0]
        pnm = src.forward_half_pft(
            r,
            Pm.real,
            Pm.imag,
            self.knm_flat,
            self.Nnm_flat,
            self.m2d_flat,
            self.Nr,
            self.Np,
        )
        pnm = pnm.reshape((self.Nr * self.Np, 2))
        Pnm = pnm[:, 0] + 1j * pnm[:, 1]
        Pnm = Pnm.reshape(np.shape(self.m2d))
        return Pnm

    def _benchmark_backward(self, Pnm: np.ndarray) -> np.ndarray:
        """
        Computes the backwards polar Fourier transform. meaning this computes
        the transform with the double integral and no extra optimisations.

        Parameters
        ----------
        Pnm : 2darray
            Polar fourier coefficients.

        Returns
        -------
        f : 2darray
            Field values on a 2D polar grid.
        """
        Pnm = Pnm.flatten()
        f = np.zeros(np.shape(self.r2d))
        for i in range(0, len(self.m2d_flat)):
            # Only compute real component of Pnm*Psi
            # Real: Pnm*Psi = Psi.real * f.real + (i^2) * Psi.imag * f.imag
            # Imag: Pnm*Psi = i * Psi.real * f.imag + i * Psi.imag * f.real
            Psi = pft_basis.get_Psi_nm(
                self.n2d_flat[i],
                self.m2d_flat[i],
                self.r2d,
                self.p2d,
                self.knm_flat[i],
                self.Nnm_flat[i],
            )
            f += Pnm.real[i] * Psi.real - Pnm.imag[i] * Psi.imag
            utils.progress_bar(i, len(self.m2d_flat))
        return f

    def _half_fft_backward(self, Pnm: np.ndarray) -> np.ndarray:
        """
        Computes the polar Fourier coefficients using FFT for the angular components
        which means each coefficient is calculated from a single integral rather than a
        double integral.

        Parameters
        ----------
        Pnm : 2darray
            Polar fourier coefficients.

        Returns
        -------
        f : 2darray
            Field values on a 2D polar grid.
        """
        Pm = np.zeros(np.shape(self.m2d)) + 1j * np.zeros(np.shape(self.m2d))
        r = self.r2d[0]
        for i in range(0, len(self.m2d_flat)):
            Rnm = pft_basis.get_Rnm(
                r, self.m2d_flat[i], self.knm_flat[i], self.Nnm_flat[i]
            )
            if self.m2d_flat[i] < 0:
                m_index = self.Np + self.m2d_flat[i]
            else:
                m_index = self.m2d_flat[i]
            n_index = self.n2d_flat[i] - 1
            Pm.real[m_index] += Rnm * Pnm.real[m_index, n_index]
            Pm.imag[m_index] += Rnm * Pnm.imag[m_index, n_index]
            utils.progress_bar(i, len(self.m2d_flat))
        f = cart.ifft1D(Pm, 2.0 * np.pi, axis=0)
        return f

    def _half_fft_half_numba_backward(self, Pnm: np.ndarray) -> np.ndarray:
        """
        Computes the polar Fourier coefficients using FFT for the angular components
        which means each coefficient is calculated from a single integral rather than a
        double integral.

        Parameters
        ----------
        Pnm : 2darray
            Polar fourier coefficients.

        Returns
        -------
        f : 2darray
            Field values on a 2D polar grid.
        """
        r = self.r2d[0]
        Pnm = Pnm.flatten()
        pm = src.backward_half_pft(
            r,
            Pnm.real,
            Pnm.imag,
            self.knm_flat,
            self.Nnm_flat,
            self.m2d_flat,
            self.n2d_flat,
            self.Nr,
            self.Np,
        )
        pm = pm.reshape((self.Nr * self.Np, 2))
        Pm = pm[:, 0] + 1j * pm[:, 1]
        Pm = Pm.reshape(np.shape(self.m2d))
        f = cart.ifft1D(Pm, 2.0 * np.pi, axis=0)
        return f

    def forward(self, f: np.ndarray, method: str = "half_fft_half_numba") -> np.ndarray:
        """Computes the forward PFT. Use method='half_fft_half_numba' for fatest calculation.

        Parameters
        ----------
        f : 2darray
            Field values on a 2D polar grid.
        method : str, optional
            Method for computing the PFT:
                - 'benchmark' : slow calculation with no optimisations for testing and benchmarking.
                - 'half_fft' : uses FFT for the angular components for faster computation.
                - 'half_fft_half_numba' : Like the above but uses fortran source code to do the other half.

        Returns
        -------
        Pnm : 2darray
            Polar fourier coefficients.
        """
        if method == "benchmark":
            Pnm = self._benchmark_forward(f)
        elif method == "half_fft":
            Pnm = self._half_fft_forward(f)
        elif method == "half_fft_half_numba":
            Pnm = self._half_fft_half_numba_forward(f)
        return Pnm

    def backward(self, Pnm: np.ndarray, method="half_fft_half_numba") -> np.ndarray:
        """Computes the forward PFT. Use method='half_fft_half_numba' for fatest calculation.

        Parameters
        ----------
        Pnm : 2darray
            Polar fourier coefficients.
        method : str, optional
            Method for computing the PFT:
                - 'benchmark' : slow calculation with no optimisations for testing and benchmarking.
                - 'half_fft' : uses FFT for the angular components for faster computation.
                - 'half_fft_half_numba' : Like the above but uses fortran source code to do the other half.

        Returns
        -------
        f : 2darray
            Field values on a 2D polar grid.
        """
        if method == "benchmark":
            f = self._benchmark_backward(Pnm)
        elif method == "half_fft":
            f = self._half_fft_backward(Pnm)
        elif method == "half_fft_half_numba":
            f = self._half_fft_half_numba_backward(Pnm)
        return f

    def clean(self) -> None:
        """
        Reinitialises the class.
        """
        self.__init__()
