# import numpy as np
# import pytest

# from shift.cart import (
#     slab_fft1D, slab_ifft1D,
#     slab_dct1D, slab_idct1D,
#     slab_dst1D, slab_idst1D,
#     _get_splits_subset_2D,
#     _get_empty_split_array_2D, _get_empty_split_array_3D
# )


# # --------------------------
# # FFT/DCT/DST roundtrip tests
# # --------------------------

# @pytest.mark.parametrize("func_fwd, func_inv", [
#     (slab_fft1D, slab_ifft1D),
#     (slab_dct1D, slab_idct1D),
#     (slab_dst1D, slab_idst1D),
# ])
# def test_roundtrip_transforms(func_fwd, func_inv):
#     ngrid = 16
#     boxsize = 2.0
#     rng = np.random.default_rng(42)
#     f_real = rng.normal(size=ngrid)

#     f_freq = func_fwd(f_real, boxsize, ngrid)
#     f_back = func_inv(f_freq, boxsize, ngrid)

#     # Should reconstruct the original signal
#     np.testing.assert_allclose(f_back, f_real, rtol=1e-10, atol=1e-10)


# def test_fft_inverse_on_axis():
#     # 2D data, FFT along axis 0
#     arr = np.random.rand(8, 4)
#     boxsize = 1.0
#     ngrid = arr.shape[0]

#     f = slab_fft1D(arr, boxsize, ngrid, axis=0)
#     back = slab_ifft1D(f, boxsize, ngrid, axis=0)

#     np.testing.assert_allclose(back, arr, rtol=1e-10)


# # --------------------------
# # Split subset tests
# # --------------------------

# def test_get_splits_subset_reverse_and_normal():
#     xs1 = np.array([0, 5, 10])
#     xs2 = np.array([5, 10, 15])
#     ys1 = np.array([0, 7, 14])
#     ys2 = np.array([7, 14, 21])

#     r1, r2 = 1, 2

#     # Normal order
#     res_normal = _get_splits_subset_2D(r1, r2, xs1, xs2, ys1, ys2, reverse=False)
#     assert res_normal == (xs1[r2], xs2[r2], ys1[r1], ys2[r1])

#     # Reverse order
#     res_reverse = _get_splits_subset_2D(r1, r2, xs1, xs2, ys1, ys2, reverse=True)
#     assert res_reverse == (xs1[r1], xs2[r1], ys1[r2], ys2[r2])


# # --------------------------
# # Split array generators
# # --------------------------

# def test_get_empty_split_array_2D_real_and_complex():
#     xs1, xs2 = np.array([0, 5]), np.array([5, 10])
#     ys1, ys2 = np.array([0, 4]), np.array([4, 8])

#     arr_real = _get_empty_split_array_2D(xs1, xs2, ys1, ys2, rank=0, axis=0, iscomplex=False)
#     assert arr_real.shape == (5, 8)
#     assert np.all(arr_real == 0.0)

#     arr_cplx = _get_empty_split_array_2D(xs1, xs2, ys1, ys2, rank=0, axis=0, iscomplex=True)
#     assert arr_cplx.shape == (5, 8)
#     assert np.iscomplexobj(arr_cplx)


# def test_get_empty_split_array_3D_real_and_complex():
#     xs1, xs2 = np.array([0, 3]), np.array([3, 6])
#     ys1, ys2 = np.array([0, 2]), np.array([2, 4])
#     zs1, zs2 = np.array([0, 5]), np.array([5, 10])

#     arr_real = _get_empty_split_array_3D(xs1, xs2, ys1, ys2, zs1, zs2, rank=0, axis=0, iscomplex=False)
#     assert arr_real.shape == (3, 4, 10)
#     assert np.all(arr_real == 0.0)

#     arr_cplx = _get_empty_split_array_3D(xs1, xs2, ys1, ys2, zs1, zs2, rank=0, axis=0, iscomplex=True)
#     assert arr_cplx.shape == (3, 4, 10)
#     assert np.iscomplexobj(arr_cplx)


# def test_invalid_axis_errors():
#     xs1, xs2 = np.array([0, 5]), np.array([5, 10])
#     ys1, ys2 = np.array([0, 4]), np.array([4, 8])
#     zs1, zs2 = np.array([0, 2]), np.array([2, 4])

#     with pytest.raises(AssertionError):
#         _get_empty_split_array_2D(xs1, xs2, ys1, ys2, rank=0, axis=3)

#     with pytest.raises(AssertionError):
#         _get_empty_split_array_3D(xs1, xs2, ys1, ys2, zs1, zs2, rank=0, axis=5)
