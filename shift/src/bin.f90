
subroutine binbyindex(ind, weights, indlength, binlength, bins)

  ! Bins weights according to the bin index.
  !
  ! Parameters
  ! ----------
  ! ind : int
  !   Index of the bin for each weight.
  ! weights : float
  !   Weights to be binned.
  ! indlength : int
  !   Length of the index array.
  ! binlength : int
  !   Length of the bin array.
  !
  ! Returns
  ! -------
  ! bins : array
  !   The output bins.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare parameters

  integer, intent(in) :: indlength, binlength
  integer, intent(in) :: ind(indlength)
  real(kind=dp), intent(in) :: weights(indlength)
  real(kind=dp), intent(out) :: bins(binlength)

  integer :: i, i0

  ! main
  
  do i = 1, binlength
    bins(i) = 0.
  end do

  do i = 1, indlength
    i0 = ind(i) + 1
    bins(i0) = bins(i0) + weights(i)
  end do

end subroutine binbyindex
