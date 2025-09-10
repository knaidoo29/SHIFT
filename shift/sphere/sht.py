import numpy as np
import healpy as hp

from typing import Optional, Tuple

from . import grid


class SHT:
    """
    Class for computing Spherical Harmonics using Healpy.
    """

    def __init__(self) -> None:
        self.nside = None
        self.pixelisation = None
        self.Nside = None
        self.Nphi = None
        self.Ntheta = None
        self.lmax = None
        self.l = None
        self.m = None
        self.p = None
        self.t = None

    def use_healpix(self, Nside: int) -> None:
        """
        Define maps in Healpix format.

        Parameters
        ----------
        Nside : int
            Nside of the healpix map.
        """
        self.pixelisation = "healpix"
        self.Nside = Nside
        self.t, self.p = hp.pix2ang(self.Nside, np.arange(hp.nside2npix(self.Nside)))

    def use_grid(self, Nphi: int, Ntheta: Optional[int] = None) -> None:
        """
        Define maps in Longitude Latitude format.

        Parameters
        ----------
        Nphi : int
            Number of divisions along the longitude direction, must be even.
        Ntheta : int, optional
            Number of divisions along the latitude directions, if None is given
            Ntheta = Nphi/2.
        """
        self.p2d, self.t2d = grid.uspheregrid(Nphi, Ntheta)
        self.pixelisation = "lonlat_grid"
        self.Nphi = Nphi
        if Ntheta is None:
            self.Ntheta = int(Nphi / 2)
        else:
            self.Ntheta = Ntheta

    def _forward_healpy(self, f: np.ndarray, lmax: int) -> np.ndarray:
        """
        Forward Spherical Harmonic Transform by healpix.

        Parameters
        ----------
        f : array
            Healpix map.
        lmax : int, optional
            Maximum l to perform the SHT.
        """
        alm = hp.map2alm(f, lmax=lmax)
        self.lmax = hp.Alm.getlmax(len(alm))
        self.l, self.m = hp.Alm.getlm(self.lmax)
        return alm

    def forward(self, f: np.ndarray, lmax: Optional[int] = None) -> np.ndarray:
        """
        Forward Spherical Harmonic Transform.

        Parameters
        ----------
        f : array
            Healpix map.
        lmax : int, optional
            Maximum l to perform the SHT.
        """
        if self.pixelisation == "healpix":
            assert self.Nside == hp.npix2nside(
                len(f)
            ), "Input map does not match the given Nside."
            alm = self._forward_healpy(f, lmax)
        else:
            assert False, "Only healpix pixelisation is supported at the moment."
        return alm

    def _backward_healpy(
        self,
        alm: np.ndarray,
        nside: Optional[int] = None,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward Spherical Harmonic Transform with healpix.

        Parameters
        ----------
        alm : complex
            Spherical harmonic coefficients.
        nside : int, optional
            Nside of an input healpix map.
        lmax : int, optional
            If you want to specify the map making to a given lmax.

        Returns
        -------
        f : array
            Healpix map.
        """
        if Nside is None:
            Nside = self.Nside
        f = hp.alm2map(alm, Nside, lmax=lmax, mmax=mmax)
        return f

    def backward(
        self,
        alm: np.ndarray,
        nside: Optional[int] = None,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward Spherical Harmonic Transform.

        Parameters
        ----------
        alm : complex
            Spherical harmonic coefficients.
        nside : int, optional
            Nside of an input healpix map.
        lmax : int, optional
            If you want to specify the map making to a given lmax.

        Returns
        -------
        f : array
            Healpix map.
        """
        if self.pixelisation == "healpix":
            return self._backward_healpy(alm, nside=nside, lmax=lmax, mmax=mmax)
        else:
            assert False, "Only healpix pixelisation is supported at the moment."

    def clean(self) -> None:
        """
        Reinitialises the class.
        """
        self.__init__()
