import numpy as np
import healpy as hp

from . import grid


class SHT:

    """Class for computing Spherical Harmonics using Healpy."""

    def __init__(self):
        self.nside = None
        self.pixelisation = None
        self.Nside = None
        self.Nphi = None
        self.Ntheta = None
        self.lmax = None


    def use_healpix(self, Nside):
        """Define maps in Healpix format.

        Parameters
        ----------
        Nside : int
            Nside of the healpix map.
        """
        self.pixelisation = 'healpix'
        self.Nside = Nside


    def use_grid(self, Nphi, Ntheta=None):
        """Define maps in Longitude Latitude format.

        Parameters
        ----------
        Nphi : int
            Number of divisions along the longitude direction, must be even.
        Ntheta : int, optional
            Number of divisions along the latitude directions, if None is given
            Ntheta = Nphi/2.
        """
        self.p2d, self.t2d = grid.uspheregrid(Nphi, Ntheta)
        self.pixelisation = 'lonlat_grid'
        self.Nphi = Nphi
        if Ntheta is None:
            self.Ntheta = int(Nphi/2)
        else:
            self.Ntheta = Ntheta


    def forward(self, f, lmax=None)
