import numpy as np
import pytest
from shift.cart import grid1D, grid2D, grid3D  # replace with actual module name


# --------------------
# grid1D Tests
# --------------------

def test_grid1D_basic():
    boxsize, ngrid = 10.0, 5
    xedges, x = grid1D(boxsize, ngrid)
    assert len(xedges) == ngrid + 1
    assert len(x) == ngrid
    # Check uniform spacing
    np.testing.assert_allclose(np.diff(xedges), boxsize / ngrid)
    # Check centers
    np.testing.assert_allclose(x, (xedges[:-1] + xedges[1:]) / 2)


def test_grid1D_origin_shift():
    boxsize, ngrid, origin = 4.0, 4, 2.0
    xedges, x = grid1D(boxsize, ngrid, origin)
    assert np.isclose(xedges[0], origin)
    assert np.isclose(x[0], origin + boxsize/ngrid/2)


# --------------------
# grid2D Tests
# --------------------

def test_grid2D_square_box_scalar():
    boxsize, ngrid = 6.0, 3
    x2D, y2D = grid2D(boxsize, ngrid)
    assert x2D.shape == (ngrid, ngrid)
    assert y2D.shape == (ngrid, ngrid)
    # Spacing should be uniform
    dx = np.unique(np.diff(x2D[:, 0]))
    dy = np.unique(np.diff(y2D[0, :]))
    np.testing.assert_allclose(dx, boxsize / ngrid)
    np.testing.assert_allclose(dy, boxsize / ngrid)


def test_grid2D_rectangular_box_list():
    boxsize = [4.0, 6.0]
    ngrid = [2, 3]
    x2D, y2D = grid2D(boxsize, ngrid, origin=[1.0, -1.0])
    assert x2D.shape == (2, 3)
    assert y2D.shape == (2, 3)
    # Check origin offsets
    assert np.isclose(x2D[0, 0], 1.0 + (boxsize[0]/ngrid[0])/2)
    assert np.isclose(y2D[0, 0], -1.0 + (boxsize[1]/ngrid[1])/2)


# --------------------
# grid3D Tests
# --------------------

def test_grid3D_cube_box_scalar():
    boxsize, ngrid = 9.0, 3
    x3D, y3D, z3D = grid3D(boxsize, ngrid)
    assert x3D.shape == (ngrid, ngrid, ngrid)
    assert y3D.shape == (ngrid, ngrid, ngrid)
    assert z3D.shape == (ngrid, ngrid, ngrid)
    # Uniform spacing
    dx = np.unique(np.diff(x3D[:, 0, 0]))
    np.testing.assert_allclose(dx, boxsize / ngrid)


def test_grid3D_rectangular_box_list():
    boxsize = [2.0, 4.0, 6.0]
    ngrid = [2, 3, 4]
    origin = [0.5, -0.5, 1.0]
    x3D, y3D, z3D = grid3D(boxsize, ngrid, origin)
    assert x3D.shape == (2, 3, 4)
    assert y3D.shape == (2, 3, 4)
    assert z3D.shape == (2, 3, 4)
    # First center positions respect origins
    assert np.isclose(x3D[0, 0, 0], origin[0] + boxsize[0]/ngrid[0]/2)
    assert np.isclose(y3D[0, 0, 0], origin[1] + boxsize[1]/ngrid[1]/2)
    assert np.isclose(z3D[0, 0, 0], origin[2] + boxsize[2]/ngrid[2]/2)

