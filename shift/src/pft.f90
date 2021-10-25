
subroutine get_rnm(r, m, knm, nnm, lenr, rnm)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr
  real(kind=dp), intent(in) :: r(lenr), knm, nnm
  integer, intent(in) :: m
  real(kind=dp), intent(out) :: rnm(lenr)

  integer :: i

  do i = 1, lenr
    rnm(i) = (1./SQRT(Nnm))*BESSEL_JN(m, knm*r(i))
  end do

end subroutine get_rnm


subroutine forward_half_pft(r, pm_real, pm_imag, knm_flat, nnm_flat, m2d_flat, lenr, lenp, pnm)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr, lenp
  real(kind=dp), intent(in) :: r(lenr)
  real(kind=dp), intent(in) :: pm_real(lenp*lenr), pm_imag(lenp*lenr)
  real(kind=dp), intent(in) :: knm_flat(lenp*lenr), nnm_flat(lenp*lenr)
  integer, intent(in) :: m2d_flat(lenp*lenr)
  real(kind=dp), intent(out) :: pnm(2*lenp*lenr)

  integer :: i, j, m_index, pm_index
  real(kind=dp) :: dr, rnm(lenr)

  dr = r(2) - r(1)

  do i = 1, 2*lenr*lenp
    pnm(i) = 0.
  end do

  do i = 1, lenr*lenp

    call get_rnm(r, m2d_flat(i), knm_flat(i), nnm_flat(i), lenr, rnm)

    if (m2d_flat(i) < 0) then
      m_index = lenp + m2d_flat(i)
    else
      m_index = m2d_flat(i)
    end if

    do j = 1, lenr

      pm_index = m_index*lenr + j
      pnm(2*i-1) = pnm(2*i-1) + r(j)*rnm(j)*pm_real(pm_index)*dr
      pnm(2*i) = pnm(2*i) + r(j)*rnm(j)*pm_imag(pm_index)*dr

    end do

  end do

end subroutine forward_half_pft


subroutine backward_half_pft(r, pnm_real, pnm_imag, knm_flat, nnm_flat, m2d_flat, n2d_flat, lenr, lenp, pm)

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lenr, lenp
  real(kind=dp), intent(in) :: r(lenr)
  real(kind=dp), intent(in) :: pnm_real(lenp*lenr), pnm_imag(lenp*lenr)
  real(kind=dp), intent(in) :: knm_flat(lenp*lenr), nnm_flat(lenp*lenr)
  integer, intent(in) :: m2d_flat(lenp*lenr), n2d_flat(lenp*lenr)
  real(kind=dp), intent(out) :: pm(2*lenp*lenr)

  integer :: i, j, m_index, n_index, pm_index, pnm_index
  real(kind=dp) :: dr, rnm(lenr)

  dr = r(2) - r(1)

  do i = 1, 2*lenr*lenp
    pm(i) = 0.
  end do

  do i = 1, lenr*lenp

    call get_rnm(r, m2d_flat(i), knm_flat(i), nnm_flat(i), lenr, rnm)

    if (m2d_flat(i) < 0) then
      m_index = lenp + m2d_flat(i)
    else
      m_index = m2d_flat(i)
    end if
    n_index = n2d_flat(i) - 1

    do j = 1, lenr

      pm_index = m_index*lenr + j
      pnm_index = m_index*lenr + n_index + 1

      pm(2*pm_index-1) = pm(2*pm_index-1) + rnm(j)*pnm_real(pnm_index)
      pm(2*pm_index) = pm(2*pm_index) + rnm(j)*pnm_imag(pnm_index)

    end do

  end do

end subroutine backward_half_pft
